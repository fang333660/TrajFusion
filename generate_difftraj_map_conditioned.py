# generate.py (V_FINAL_UNIFIED - 最终统一版，适配所有实验场景)

import torch
import yaml
import argparse
import os
import numpy as np
import pandas as pd
from types import SimpleNamespace
import time
from pathlib import Path
from typing import Optional, Any
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import networkx as nx
import osmnx as ox
import json
import sys
import traceback


# --- 动态导入自定义模块的辅助函数 ---
def import_modules_from_config(config: SimpleNamespace) -> tuple[Any, Any]:
    """根据配置文件动态导入正确的模型和扩散模块。"""
    use_lra = getattr(config.model, 'use_local_road_awareness', False)
    g2_loss_enabled = getattr(config.training, 'endpoint_loss_weight', 0.0) > 0 or \
                      getattr(config.training, 'geom_loss_weight', 0.0) > 0

    if use_lra:
        # 这种情况对应你的完整版模型和“无G²Diff”消融版
        unet_module_name = "Traj_UNet"  # 两个版本都使用完整的UNet
        if g2_loss_enabled:
            diffusion_module_name = "GaussianDiffusion"
            print("信息: 将加载 [完整版] 模型模块 (LRA + G²Diff)。")
        else:
            diffusion_module_name = "GaussianDiffusion"
            print("信息: 将加载 [消融版-无G²] 模型模块 (保留LRA, 无G²Diff)。")
    else:
        # 这种情况对应你的“无LRA”消融版和基线版
        if g2_loss_enabled:
            unet_module_name = "Traj_UNet_Ablation_NoLRA"
            diffusion_module_name = "GaussianDiffusion_Ablation_NoLRA"
            print("信息: 将加载 [消融版-无LRA] 模型模块 (无LRA, 保留G²Diff)。")
        else:
            unet_module_name = "Traj_UNet_Baseline"
            diffusion_module_name = "GaussianDiffusion_Baseline"
            print("信息: 将加载 [基线版] 模型模块 (无LRA, 无G²Diff)。")

    try:
        # 确保 utils 目录在Python的搜索路径中
        utils_path = str(Path(__file__).resolve().parent / 'utils')
        if utils_path not in sys.path:
            sys.path.insert(0, utils_path)

        unet_module = __import__(unet_module_name)
        Guide_UNet = unet_module.Guide_UNet

        diffusion_module = __import__(diffusion_module_name)
        GaussianDiffusion = diffusion_module.GaussianDiffusion

        return Guide_UNet, GaussianDiffusion
    except ImportError as e:
        print(f"\n错误: 动态导入模块 '{unet_module_name}' 或 '{diffusion_module_name}' 时失败。")
        print(f"请确保对应的.py文件存在于 'utils' 文件夹中。")
        print(f"详细信息: {e}\n")
        traceback.print_exc()
        exit(1)


def denormalize_coord(norm_coord, min_val, max_val):
    """将 [-1, 1] 范围的坐标反归一化到原始值。"""
    norm_coord = np.asarray(norm_coord)
    norm_01 = (np.clip(norm_coord, -1.0, 1.0) + 1.0) / 2.0
    return norm_01 * (max_val - min_val) + min_val


def dict_to_ns(d: dict) -> SimpleNamespace:
    """将字典递归转换为 SimpleNamespace 对象。"""
    if not isinstance(d, dict): return d
    ns = SimpleNamespace();
    [setattr(ns, k, dict_to_ns(v) if isinstance(v, dict) else v) for k, v in d.items()];
    return ns


def generate_trajectories(
        checkpoint_path: str,
        config_path: str,
        num_samples: int,
        output_csv: str,
        device: str = 'cuda',
        batch_size: Optional[int] = None,
        guidance_scale: Optional[float] = None,
):
    start_time_script = time.time()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"--- 初始化 TrajFusion 生成脚本 (最终统一版) ---")
    print(f"使用设备: {device}")

    # --- 1. 加载配置 ---
    print("\n--- 1. 加载配置和检查点 ---")
    with open(config_path, 'r', encoding='utf-8') as f:
        current_config_dict = yaml.safe_load(f)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 优先加载检查点中的配置来构建模型，确保结构一致性
    if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
        config_dict = checkpoint['config']
        # 但保留当前配置文件中的数据路径和采样设置
        config_dict['data'] = current_config_dict['data']
        config_dict['sampling'] = current_config_dict['sampling']
        config = dict_to_ns(config_dict)
        print("信息: 模型结构和扩散参数已从检查点内的配置加载。")
    else:
        config = dict_to_ns(current_config_dict)
        print("警告: 检查点中未找到配置信息，将使用传入的config文件。")

    model_state_to_load = checkpoint.get('ema_model_state_dict', checkpoint.get('model_state_dict'))
    print("信息: 正在加载 {} 模型权重。".format("EMA" if 'ema_model_state_dict' in checkpoint else "标准"))

    # --- 2. 动态导入正确的模块 ---
    Guide_UNet, GaussianDiffusion = import_modules_from_config(config)

    # --- 3. 提取参数 ---
    data_dir = Path(config.data.base_data_dir)
    target_len = config.data.traj_length
    in_channels = config.model.in_channels
    batch_size_to_use = batch_size if batch_size is not None else config.sampling.batch_size
    guidance_scale_to_use = guidance_scale if guidance_scale is not None else config.sampling.guidance_scale_sampling

    stats_path = data_dir / "norm_stats.json"
    if not stats_path.exists():
        print(f"错误: 归一化统计文件未找到: {stats_path}");
        exit(1)
    with open(stats_path, 'r') as f:
        norm_stats = json.load(f)
    coord_stats = norm_stats['coordinate_normalization']
    lat_min, lat_max, lon_min, lon_max = coord_stats['lat_min'], coord_stats['lat_max'], coord_stats['lon_min'], \
    coord_stats['lon_max']

    # --- 4. 初始化模型 ---
    print("\n--- 4. 初始化模型和扩散处理器 ---")
    use_lra = getattr(config.model, 'use_local_road_awareness', False)
    model_init_kwargs = {'config': config}

    if use_lra:
        print("LRA已启用，正在加载OSM图...")
        try:
            osm_graph = ox.load_graphml(config.data.osm_graph_path)
            model_init_kwargs['osm_graph_data'] = nx.convert_node_labels_to_integers(osm_graph, first_label=0,
                                                                                     ordering='default')
            print("OSM图加载成功。")
        except Exception as e:
            print(f"错误: 加载OSM图失败: {e}。");
            exit(1)

    model = Guide_UNet(**model_init_kwargs).to(device)
    model.load_state_dict(model_state_to_load);
    model.eval()
    diffusion = GaussianDiffusion(config, model, device=device)
    print("模型和 Diffusion 初始化完成。")

    # --- 5. 加载引导数据 ---
    print("\n--- 5. 加载引导数据 ---")
    cond_path = data_dir / config.data.test_cond_file
    all_conditions_np = np.load(cond_path)

    # 我们将生成固定长度的轨迹，后续可以根据需要进行分析或截断
    all_real_lengths_np = np.full(len(all_conditions_np), target_len)

    all_lra_features = None
    if use_lra:
        lra_file_name = config.data.test_lra_file
        lra_path = data_dir / lra_file_name
        if lra_path.exists():
            print(f"LRA已启用, 正在加载LRA特征: {lra_path}")
            if lra_path.suffix == '.npy':
                all_lra_features = torch.from_numpy(np.load(lra_path, mmap_mode='r')).float()
            elif lra_path.suffix == '.pt':
                all_lra_features = torch.load(lra_path, map_location='cpu', weights_only=True)
        else:
            print(f"警告: LRA已启用, 但LRA文件 '{lra_path}' 未找到。将无LRA引导。");
            use_lra = False

    num_available = len(all_conditions_np)
    # 从测试集中随机抽取指定的样本数量用于生成
    indices = np.random.choice(num_available, num_samples, replace=(num_samples > num_available))
    conditions_to_gen = torch.from_numpy(all_conditions_np[indices]).float()

    dataset_tensors = [conditions_to_gen]
    if use_lra and all_lra_features is not None:
        dataset_tensors.append(all_lra_features[indices].clone())  # .clone()很重要

    dataset = TensorDataset(*dataset_tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size_to_use, shuffle=False)
    print(f"已准备 {num_samples} 个条件用于生成。")

    # --- 6. 开始生成 ---
    print(f"\n--- 6. 开始生成 {num_samples} 条轨迹 (引导强度: {guidance_scale_to_use}) ---")
    all_generated_dfs = []
    total_generated = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="生成批次")
        for i, batch in enumerate(pbar):
            lra_batch = None
            if use_lra and len(batch) == 2:
                cond_batch, lra_batch = batch
                lra_batch = lra_batch.to(device)
            else:
                cond_batch = batch[0]

            cond_batch = cond_batch.to(device)
            shape = (cond_batch.shape[0], in_channels, target_len)

            # 动态构建参数字典，以适配不同版本的 p_sample_loop
            sample_loop_kwargs = {
                'shape': shape,
                'attributes': cond_batch,
                'guidance_scale_sampling': guidance_scale_to_use
            }
            if use_lra and lra_batch is not None:
                sample_loop_kwargs['precomputed_lra_for_sampling_run'] = lra_batch

            # 使用 **kwargs 解包字典来调用函数，安全又灵活
            generated_trajs = diffusion.p_sample_loop(**sample_loop_kwargs)

            gen_np = generated_trajs.cpu().numpy()
            batch_start_index = i * batch_size_to_use

            for j in range(gen_np.shape[0]):
                current_index_in_dataset = indices[batch_start_index + j]
                real_length = int(all_real_lengths_np[current_index_in_dataset])
                if real_length <= 1: continue

                traj_norm = gen_np[j, :, :real_length]
                # 假设 Channel 0 是 lat, Channel 1 是 lon
                lat_denorm = denormalize_coord(traj_norm[0, :], lat_min, lat_max)
                lon_denorm = denormalize_coord(traj_norm[1, :], lon_min, lon_max)

                df = pd.DataFrame({'traj_id': f'gen_{total_generated}', 'lat': lat_denorm, 'lon': lon_denorm})
                all_generated_dfs.append(df)
                total_generated += 1

    # --- 7. 保存结果 ---
    print(f"\n--- 7. 保存结果 ---")
    if not all_generated_dfs:
        print("警告: 没有生成任何有效的轨迹。")
    else:
        final_df = pd.concat(all_generated_dfs, ignore_index=True)
        output_file_path = Path(output_csv)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_file_path, index=False)
        print(f"轨迹已成功保存到: {output_file_path.resolve()}")

    print(f"\n--- 生成脚本执行完毕 (总耗时: {time.time() - start_time_script:.2f} 秒) ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用 TrajFusion 模型生成轨迹 (最终统一版)。")
    parser.add_argument('--checkpoint', type=str, required=True, help="模型检查点 (.pt) 文件路径。")
    parser.add_argument('--config_path', type=str, required=True, help="与检查点匹配的YAML配置文件路径。")
    parser.add_argument('--num_samples', type=int, required=True, help="要生成的轨迹数量。")
    parser.add_argument('--output_csv', type=str, required=True, help="保存生成轨迹的CSV文件路径。")
    parser.add_argument('--batch_size', type=int, default=None, help="覆盖config中的批处理大小。")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="使用的设备。")
    parser.add_argument('--guidance_scale', type=float, default=None, help="覆盖config中的引导强度。")
    args = parser.parse_args()

    generate_trajectories(
        checkpoint_path=args.checkpoint,
        config_path=args.config_path,
        num_samples=args.num_samples,
        output_csv=args.output_csv,
        device=args.device,
        batch_size=args.batch_size,
        guidance_scale=args.guidance_scale
    )