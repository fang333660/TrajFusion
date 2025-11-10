# main.py (最终净化版)
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import math
import datetime
import os
from torch.utils.data import DataLoader, Dataset
from types import SimpleNamespace
import yaml
from pathlib import Path
import argparse
from tqdm import tqdm
import gc
import traceback
import sys
import inspect
import importlib
import networkx as nx

# --- 模块重载和导入 ---
print("--- Python Module Search Path (sys.path) ---")
for p in sys.path: print(p)
print("-------------------------------------------")
try:
    from utils import GaussianDiffusion as GD_module_ref;

    importlib.reload(GD_module_ref)
    from utils.GaussianDiffusion import GaussianDiffusion

    print("GaussianDiffusion reloaded successfully.")
except Exception as e:
    print(f"Could not force reload GaussianDiffusion: {e}");
    from utils.GaussianDiffusion import GaussianDiffusion
try:
    from utils import Traj_UNet as TU_module_ref;

    importlib.reload(TU_module_ref)
    from utils.Traj_UNet import Guide_UNet

    print("Traj_UNet reloaded successfully.")
except Exception as e:
    print(f"Could not force reload Traj_UNet: {e}");
    from utils.Traj_UNet import Guide_UNet

print(f"DEBUG: GaussianDiffusion loaded from: {inspect.getfile(GaussianDiffusion)}")
print(f"DEBUG: Guide_UNet loaded from: {inspect.getfile(Guide_UNet)}")
try:
    from utils.EMA import EMAHelper
    from utils.logger import Logger
    import osmnx as ox
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
    print("All custom modules imported successfully.")
except ImportError as e:
    print(f"Error: Failed to import custom modules: {e}");
    traceback.print_exc();
    TENSORBOARD_AVAILABLE = False


# --- 数据集定义 ---
def simple_collate_fn(batch):
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch: return None
    collated = {}
    keys = valid_batch[0].keys()
    for key in keys:
        if key == 'lra_features' and any(key not in item for item in valid_batch):
            continue
        try:
            collated[key] = torch.stack([item[key] for item in valid_batch])
        except:  # For non-tensor data like traj_id
            collated[key] = [item[key] for item in valid_batch]
    return collated


class NpyFriendlyLoaderDataset(Dataset):
    def __init__(self, coord_file_path: str, cond_file_path: str, lra_features_file_path: Optional[str] = None):
        super().__init__()
        print(f"NpyFriendlyLoaderDataset Initializing...")
        print(f"  Loading coords: {coord_file_path}")
        self.coord_tensor = torch.load(coord_file_path, map_location='cpu', weights_only=True)
        print(f"  Loading conditions: {cond_file_path}")
        self.cond_array = np.load(cond_file_path, allow_pickle=True)
        self.num_samples = self.coord_tensor.shape[0]
        self.lra_features_data = None
        if lra_features_file_path:
            p = Path(lra_features_file_path)
            if p.exists() and p.suffix == '.npy':
                print(f"  Memory-mapping LRA features: {p}")
                self.lra_features_data = np.load(p, mmap_mode='r', allow_pickle=True)
                assert self.lra_features_data.shape[0] == self.num_samples
        print(f"Dataset ready with {self.num_samples} samples.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = {
            'trajectory': self.coord_tensor[idx],
            'attributes': torch.from_numpy(self.cond_array[idx]).float(),
        }
        if self.lra_features_data is not None:
            item['lra_features'] = torch.from_numpy(np.copy(self.lra_features_data[idx])).float()
        return item


# --- 训练函数 ---
def train(config: SimpleNamespace, logger: Logger, exp_dir: Path, timestamp: str, config_dict_to_save: dict,
          unet_model: nn.Module, optimizer: torch.optim.Optimizer,
          initial_start_epoch: int = 1, initial_global_step: int = 0):
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    unet_model.to(device)
    tb_writer = SummaryWriter(log_dir=str(exp_dir / "tb_logs" / timestamp)) if TENSORBOARD_AVAILABLE else None

    # --- 数据加载 ---
    base_data_dir = Path(config.data.base_data_dir).resolve()
    lra_path = base_data_dir / f"{Path(config.data.train_lra_file).stem}.npy"
    lra_path_for_dataset = str(lra_path) if lra_path.exists() else None
    if lra_path_for_dataset: logger.info(f"Found and using LRA features: {lra_path_for_dataset}")

    dataset = NpyFriendlyLoaderDataset(
        coord_file_path=str(base_data_dir / config.data.train_traj_file),
        cond_file_path=str(base_data_dir / config.data.train_cond_file),
        lra_features_file_path=lra_path_for_dataset
    )
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True,
                            num_workers=config.data.num_workers, drop_last=True,
                            pin_memory=(device.type == "cuda"), collate_fn=simple_collate_fn)
    logger.info(f"Data loading complete. Dataset size: {len(dataset)}")

    # --- 初始化 ---
    ema_helper = EMAHelper(mu=config.model.ema_rate) if config.model.ema else None
    if ema_helper: ema_helper.register(unet_model)
    diffusion_process = GaussianDiffusion(config, unet_model, device=device)
    model_run_save_dir = exp_dir / 'models' / timestamp
    model_run_save_dir.mkdir(parents=True, exist_ok=True)
    global_step = initial_global_step

    # --- 训练循环 ---
    logger.info(f"--- Starting Training (Traj-Fusion Model / G²Diff Strategy) from epoch {initial_start_epoch} ---")
    for epoch in range(initial_start_epoch, config.training.n_epochs + 1):
        logger.info(f"<---- Epoch-{epoch}/{config.training.n_epochs} ---->")
        unet_model.train()
        epoch_losses = {'total': [], 'main': [], 'endpoint': [], 'geom': []}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False, ncols=150)
        for batch_data in pbar:
            if batch_data is None: continue
            optimizer.zero_grad(set_to_none=True)
            loss_components = diffusion_process.p_losses(
                x0=batch_data['trajectory'],
                attributes=batch_data['attributes'],
                precomputed_lra_batch=batch_data.get('lra_features'),
                uncond_prob=config.training.uncond_prob_cfg
            )
            total_loss = loss_components['total_loss']
            if torch.isnan(total_loss):
                logger.warning(f"NaN loss at step {global_step}. Skipping batch.")
                continue
            total_loss.backward()
            if config.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(unet_model.parameters(), config.training.grad_clip)
            optimizer.step()
            if ema_helper: ema_helper.update(unet_model)

            for key in epoch_losses: epoch_losses[key].append(loss_components[key.split('_')[0] + '_loss'].item())
            global_step += 1

            pbar.set_postfix({k: f"{v.item():.4f}" for k, v in loss_components.items()})
            if tb_writer:
                for k, v in loss_components.items(): tb_writer.add_scalar(f'Loss/{k}', v.item(), global_step)

        avg_losses = {k: np.mean(v) if v else 0 for k, v in epoch_losses.items()}
        logger.info(
            f"Epoch {epoch} Avg Losses -> " + " | ".join([f"{k.capitalize()}: {v:.4f}" for k, v in avg_losses.items()]))
        if tb_writer:
            for k, v in avg_losses.items(): tb_writer.add_scalar(f'Loss/Epoch_Avg_{k}', v, epoch)

        if epoch % config.training.save_epoch_interval == 0 or epoch == config.training.n_epochs:
            logger.info(f"Saving model at epoch {epoch}...")
            save_obj = {'model_state_dict': unet_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'global_step': global_step, 'epoch': epoch, 'config': config_dict_to_save}
            if ema_helper:
                ema_helper.ema(unet_model)
                save_obj['ema_model_state_dict'] = unet_model.state_dict()
                ema_helper.restore(unet_model)  # Restore original weights
            torch.save(save_obj, model_run_save_dir / f"unet_epoch_{epoch}.pt")
    logger.info("--- Training Finished ---")
    if tb_writer: tb_writer.close()


# --- 主执行块 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Traj-Fusion model.")
    parser.add_argument('--config_path', type=str, default='./utils/config_xian.yaml', help='Path to the YAML config file.')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='(Optional) Path to resume training from.')
    cmd_args = parser.parse_args()

    # --- 1. 加载配置 ---
    config_path = Path(cmd_args.config_path)
    if not config_path.is_file(): raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)


    def dict_to_ns(d):
        return SimpleNamespace(**{k: dict_to_ns(v) if isinstance(v, dict) else v for k, v in d.items()}) if isinstance(
            d, dict) else d


    config_ns = dict_to_ns(config_dict)

    # --- 2. 设置日志和实验目录 ---
    exp_base_dir = Path(config_ns.training.exp_base_dir)
    run_name = f"{config_ns.data.dataset}_TrajFusion_{datetime.datetime.now().strftime('%Y%m%d')}"
    final_exp_dir = exp_base_dir / run_name
    ts = datetime.datetime.now().strftime("%H%M%S")
    log_dir = final_exp_dir / "logs";
    log_dir.mkdir(parents=True, exist_ok=True)
    logger_instance = Logger(name="Trainer", log_path=str(log_dir / f"train_{ts}.log"), colorize=True)
    device_init = torch.device(config_ns.training.device if torch.cuda.is_available() else "cpu")

    # --- 3. 初始化模型 (最终的、干净的逻辑) ---
    osm_instance_relabeled = None
    if getattr(config_ns.model, 'use_local_road_awareness', False):
        try:
            logger_instance.info(f"Loading OSM graph from {config_ns.data.osm_graph_path}")
            osm_instance = ox.load_graphml(config_ns.data.osm_graph_path)
            osm_instance_relabeled = nx.convert_node_labels_to_integers(osm_instance, first_label=0)
            logger_instance.info("OSM graph loaded and relabeled for consistent node IDs.")
        except FileNotFoundError:
            logger_instance.error(
                f"OSM graph file not found at {config_ns.data.osm_graph_path}. LRA and Global Topology will be disabled.")

    unet_model = Guide_UNet(config_ns, osm_graph_data=osm_instance_relabeled).to(device_init)

    # --- 4. 初始化优化器和加载检查点 ---
    optimizer = torch.optim.AdamW(unet_model.parameters(), lr=config_ns.training.lr,
                                  weight_decay=getattr(config_ns.training, 'weight_decay', 0.0))
    start_epoch, global_step = 1, 0
    if cmd_args.resume_checkpoint:
        logger_instance.info(f"Resuming from checkpoint: {cmd_args.resume_checkpoint}")
        ckpt = torch.load(cmd_args.resume_checkpoint, map_location=device_init)
        unet_model.load_state_dict(ckpt.get('ema_model_state_dict', ckpt['model_state_dict']))
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        global_step = ckpt.get('global_step', 0)
        logger_instance.info(f"Resuming from epoch {start_epoch}, global step {global_step}")

    # --- 5. 开始训练 ---
    try:
        logger_instance.info("--- Effective Training Config ---")
        logger_instance.info(f"{yaml.dump(config_dict, indent=2)}")
        logger_instance.info("---------------------------------")
        train(config=config_ns, logger=logger_instance, exp_dir=final_exp_dir, timestamp=ts,
              config_dict_to_save=config_dict, unet_model=unet_model,
              optimizer=optimizer, initial_start_epoch=start_epoch, initial_global_step=global_step)
    except Exception as e:
        logger_instance.critical(f"Critical error during training: {e}")
        traceback.print_exc()
    finally:
        logger_instance.info("Training script finished.")