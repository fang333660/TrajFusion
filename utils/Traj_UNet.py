# utils/Traj_UNet.py (最终稳定版 - 完整代码)
import math
import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
import torch.nn.functional as F
import osmnx as ox
from scipy.spatial import cKDTree


# --- 1. 辅助函数和基础模块 ---
def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1D tensor"
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    return F.silu(x)


def Normalize(in_channels):
    if in_channels <= 0: return nn.Identity()
    num_groups = 32
    if num_groups > in_channels: num_groups = 1
    while in_channels % num_groups != 0 and num_groups > 1:
        num_groups //= 2
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        if with_conv: self.conv = nn.Conv1d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="linear", align_corners=False)
        if hasattr(self, 'conv'): x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        if with_conv: self.conv = nn.Conv1d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x) if hasattr(self, 'conv') else F.avg_pool1d(x, 2)


# --- 2. 注意力模块 ---
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None):
        context = context if context is not None else x
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(
            lambda t: t.view(t.shape[0], -1, self.heads, self.to_q.out_features // self.heads).transpose(1, 2),
            (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(out.shape[0], -1, self.heads * (self.to_q.out_features // self.heads))
        return self.to_out(out)


# --- 3. 核心模块 ---
class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels, dropout, temb_channels):
        super().__init__()
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, temb):
        h = self.shortcut(x)
        x = self.conv1(nonlinearity(self.norm1(x)))
        x += self.temb_proj(nonlinearity(temb))[:, :, None]
        x = self.conv2(self.dropout(nonlinearity(self.norm2(x))))
        return x + h


class AttnBlock(nn.Module):
    """
    一个将自注意力和交叉注意力结合在一起的模块，用于U-Net的瓶颈层。
    """

    def __init__(self, in_channels, context_dim):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.self_attn = Attention(in_channels)
        self.cross_attn = Attention(in_channels, context_dim)

    def forward(self, x, context):
        # Self-attention
        x = self.self_attn(self.norm(x).transpose(1, 2)).transpose(1, 2) + x
        # Cross-attention
        x = self.cross_attn(self.norm(x).transpose(1, 2), context).transpose(1, 2) + x
        return x


class Model(nn.Module):
    def __init__(self, config, context_dim):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks, dropout = config.model.num_res_blocks, config.model.dropout

        self.temb_ch = ch * 4
        self.time_embed = nn.Sequential(nn.Linear(ch, self.temb_ch), nn.SiLU(), nn.Linear(self.temb_ch, self.temb_ch))
        self.conv_in = nn.Conv1d(config.model.in_channels, ch, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        hs_channels = [ch]
        block_in = ch
        for i_level, mult in enumerate(ch_mult):
            block_out = ch * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout,
                                                    temb_channels=self.temb_ch))
                hs_channels.append(block_out)
                block_in = block_out
            if i_level != len(ch_mult) - 1:
                self.down_blocks.append(Downsample(block_in))
                hs_channels.append(block_in)

        self.mid_block = nn.ModuleList([
            ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout, temb_channels=self.temb_ch),
            AttnBlock(block_in, context_dim),
            ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout, temb_channels=self.temb_ch)
        ])

        self.up_blocks = nn.ModuleList()
        for i_level, mult in reversed(list(enumerate(ch_mult))):
            block_out = ch * mult
            for _ in range(num_res_blocks + 1):
                block_in_up = block_in + hs_channels.pop()
                self.up_blocks.append(ResnetBlock(in_channels=block_in_up, out_channels=block_out, dropout=dropout,
                                                  temb_channels=self.temb_ch))
                block_in = block_out
            if i_level != 0:
                self.up_blocks.append(Upsample(block_in))

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, out_ch, 3, padding=1)

    def forward(self, x, t, extra_embed, context):
        temb = self.time_embed(get_timestep_embedding(t, self.config.model.ch))
        global_emb = temb + extra_embed

        h = self.conv_in(x)
        hs = [h]
        for module in self.down_blocks:
            h = module(h, global_emb) if isinstance(module, ResnetBlock) else module(h)
            hs.append(h)

        h = self.mid_block[0](h, global_emb)
        h = self.mid_block[1](h, context)
        h = self.mid_block[2](h, global_emb)

        for module in self.up_blocks:
            if isinstance(module, ResnetBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, global_emb)
            else:
                h = module(h)

        return self.conv_out(nonlinearity(self.norm_out(h)))


# --- 4. 条件模块 ---
class WideAndDeep(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_time_slots, num_grid_embeddings):
        super(WideAndDeep, self).__init__()
        self.embedding_dim, self.hidden_dim, self.num_time_slots, self.num_grid_embeddings = embedding_dim, hidden_dim, num_time_slots, num_grid_embeddings
        self.wide_fc = nn.Linear(5, embedding_dim)
        self.depature_embedding = nn.Embedding(num_time_slots, hidden_dim)
        self.sid_embedding = nn.Embedding(num_grid_embeddings, hidden_dim)
        self.eid_embedding = nn.Embedding(num_grid_embeddings, hidden_dim)
        self.deep_fc1 = nn.Linear(hidden_dim * 3, embedding_dim)
        self.deep_fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.ReLU()

    def forward(self, attr):
        if attr.shape[1] != 8: raise ValueError(f"WideAndDeep expects 8 attrs, got {attr.shape[1]}")
        cont = attr[:, 1:6].float()
        dep_idx = attr[:, 0].long().clamp(0, self.num_time_slots - 1)
        sid_idx = attr[:, 6].long().clamp(0, self.num_grid_embeddings - 1)
        eid_idx = attr[:, 7].long().clamp(0, self.num_grid_embeddings - 1)
        wide_out = self.wide_fc(cont)
        cat_embed = torch.cat(
            [self.depature_embedding(dep_idx), self.sid_embedding(sid_idx), self.eid_embedding(eid_idx)], dim=1)
        deep_out = self.activation(self.deep_fc1(cat_embed))
        return wide_out + self.deep_fc2(deep_out)


class TrajFusionContextModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        lra_input_dim = config.model.num_nearest_roads_query * 15
        lra_embed_dim = 128
        self.lra_encoder = nn.Sequential(nn.Linear(lra_input_dim, 256), nn.SiLU(), nn.Linear(256, lra_embed_dim))
        print(f"  [TrajFusion] Loading global road node embeddings from: {config.data.road_embedding_path}")
        self.road_node_embeddings = torch.load(config.data.road_embedding_path, map_location='cpu', weights_only=True)
        node_embed_dim = self.road_node_embeddings.shape[1]
        self.context_dim = lra_embed_dim + node_embed_dim

    def forward(self, precomputed_lra_batch, nearest_node_ids):
        lra_embed = self.lra_encoder(precomputed_lra_batch)
        if self.road_node_embeddings.device != nearest_node_ids.device:
            self.road_node_embeddings = self.road_node_embeddings.to(nearest_node_ids.device)
        node_embed = self.road_node_embeddings[nearest_node_ids]
        return torch.cat([lra_embed, node_embed], dim=-1)


# --- 5. 主模型 ---
class Guide_UNet(nn.Module):
    def __init__(self, config: SimpleNamespace, osm_graph_data=None):
        super(Guide_UNet, self).__init__()
        self.config = config
        self.context_module = TrajFusionContextModule(config)
        self.unet = Model(config, context_dim=self.context_module.context_dim)
        temb_ch = config.model.ch * 4
        self.guide_emb = WideAndDeep(temb_ch, config.model.attr_hidden_dim, config.data.num_time_slots,
                                     config.data.num_grid_cells + 1)
        self.place_emb = WideAndDeep(temb_ch, config.model.attr_hidden_dim, config.data.num_time_slots,
                                     config.data.num_grid_cells + 1)
        if osm_graph_data:
            print("  [TrajFusion] Building node KDTree from pre-relabeled graph...")
            nodes_gdf = ox.graph_to_gdfs(osm_graph_data, edges=False, node_geometry=True)
            self.node_kdtree = cKDTree(nodes_gdf[['x', 'y']].values)
            self.node_ids_in_kdtree_order = nodes_gdf.index.to_numpy()
            print("  [TrajFusion] Node KDTree built successfully.")
        else:
            self.node_kdtree = None
            print("  [TrajFusion] Warning: No OSM graph provided.")

    def find_nearest_nodes(self, x_norm):
        lon_min, lon_max, lat_min, lat_max = self.config.data.norm.lon_min, self.config.data.norm.lon_max, self.config.data.norm.lat_min, self.config.data.norm.lat_max
        lons = ((x_norm[:, 0, :] + 1) / 2) * (lon_max - lon_min) + lon_min
        lats = ((x_norm[:, 1, :] + 1) / 2) * (lat_max - lat_min) + lat_min
        B, L = lons.shape
        query_points = np.stack([lons.flatten().detach().cpu().numpy(), lats.flatten().detach().cpu().numpy()], axis=1)
        _, indices = self.node_kdtree.query(query_points, k=1, workers=-1)
        node_ids_flat = self.node_ids_in_kdtree_order[indices]
        return torch.from_numpy(node_ids_flat).view(B, L).long().to(x_norm.device)

    def forward(self, x, t, attr, precomputed_lra_batch, uncond_prob=0.0):
        guide_attr_emb = self.guide_emb(attr)
        uncond_attr_emb = self.place_emb(torch.zeros_like(attr, device=attr.device))

        context = torch.zeros(x.shape[0], x.shape[2], self.context_module.context_dim, device=x.device)
        if self.node_kdtree is not None and precomputed_lra_batch is not None:
            nearest_node_ids = self.find_nearest_nodes(x)
            context = self.context_module(precomputed_lra_batch, nearest_node_ids)

        if self.training and uncond_prob > 0 and torch.rand(1).item() < uncond_prob:
            final_attr_emb = uncond_attr_emb
            context = torch.zeros_like(context)
        else:
            final_attr_emb = guide_attr_emb

        pred_noise = self.unet(x, t, final_attr_emb, context)
        return pred_noise