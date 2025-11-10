# visualize_trajectories_datashader.py (修正 log 错误, 显式导入 colorcet 并使用 getattr 获取 cmap)
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from PIL import Image # Import Pillow for image saving
import argparse
import os
import time
import numpy as np
import traceback
import colorcet as cc # <--- 显式导入 colorcet
import xarray as xr
import matplotlib.cm as mcm # <--- 添加或确保这行存在
# ... 其他导入 ...

def visualize_datashader(csv_path, output_png,
                         canvas_width=1000, canvas_height=1000,
                         cmap_name='fire', background_color='black',
                         log_scale=True):
    """
    使用 Datashader 可视化 CSV 文件中所有轨迹点的宏观分布。

    Args:
        csv_path (str): 包含轨迹数据的 CSV 文件路径。需要 'lat', 'lon' 列。
        output_png (str): 输出的可视化 PNG 图片文件路径。
        canvas_width (int): 输出图像的宽度（像素）。
        canvas_height (int): 输出图像的高度（像素）。
        cmap_name (str): Datashader 颜色映射名称 (e.g., 'fire', 'viridis', 'hot', 'inferno', 'bmy').
        background_color (str): 图像的背景颜色 (e.g., 'black', 'white').
        log_scale (bool): 是否对点计数应用对数缩放（有助于显示密度差异）。
    """
    print(f"--- 开始使用 Datashader 进行宏观可视化 ---")
    print(f"加载轨迹数据: {csv_path}")
    start_time = time.time()

    # --- 1. 加载数据 ---
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 文件未找到: {csv_path}")
        try:
            df_peek = pd.read_csv(csv_path, nrows=5)
        except Exception as e: print(f"错误: 无法读取 CSV 文件头部: {e}"); return
        required_cols = ['lat', 'lon']
        if not all(col in df_peek.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_peek.columns]
            raise ValueError(f"CSV 文件缺少必需列: {missing}")

        print("正在加载完整的 CSV 文件 (仅经纬度列)...")
        df = pd.read_csv(csv_path, usecols=['lat', 'lon'], low_memory=False)
        load_time = time.time(); print(f"加载完成: {len(df)} 个点 (用时: {load_time - start_time:.2f}s)")
        df = df.dropna(subset=required_cols)
        df = df[~df['lat'].isin([np.inf, -np.inf]) & ~df['lon'].isin([np.inf, -np.inf])] # 合并检查 Inf
        if df.empty: print("错误: CSV 文件加载后无有效数据。"); return
        print(f"清理后剩余有效点数: {len(df)}")
    except Exception as e: print(f"错误: 加载或处理 CSV 时出错: {e}"); traceback.print_exc(); return

    # --- 2. 准备 Datashader Canvas ---
    try:
        lon_min, lon_max = df['lon'].min(), df['lon'].max()
        lat_min, lat_max = df['lat'].min(), df['lat'].max()
        print(f"数据范围: Lon [{lon_min:.4f}, {lon_max:.4f}], Lat [{lat_min:.4f}, {lat_max:.4f}]")
        canvas = ds.Canvas(plot_width=canvas_width, plot_height=canvas_height,
                           x_range=(lon_min, lon_max), y_range=(lat_min, lat_max))

        # --- 3. 使用 Datashader 聚合点 ---
        print("正在使用 Datashader 聚合点..."); agg_start_time = time.time()
        agg = canvas.points(df, x='lon', y='lat', agg=ds.count())
        agg_time = time.time(); print(f"聚合完成 (用时: {agg_time - agg_start_time:.2f}s)")

        # --- 4. 对聚合结果进行着色 ---
        print("正在对着色聚合结果..."); shade_start_time = time.time()
        # 修正对数缩放逻辑
        if log_scale:
            agg_data = agg.values.astype(np.float64)
            np.nan_to_num(agg_data, copy=False, nan=0.0)
            agg_data_log = np.log1p(agg_data)
            agg_log = xr.DataArray(agg_data_log, coords=agg.coords, dims=agg.dims, name='count_log')
            agg_to_shade = agg_log
            print("已应用对数缩放 (log1p)。")
        else:
            agg_to_shade = agg

        # --- 修改点：使用 getattr 从 colorcet.cm 获取 cmap ---
        # --- 修改点：更稳健地获取 cmap ---
        # --- 修改点：更稳健地获取 cmap，并处理 matplotlib cmap ---
        print(f"尝试为 '{cmap_name}' 获取颜色映射...")
        selected_cmap_for_datashader = None  # 用于最终传递给 tf.shade

        try:
            # 首先尝试从 colorcet.cm 获取
            selected_cmap_for_datashader = getattr(cc.cm, cmap_name)
            print(f"成功从 colorcet.cm 获取颜色映射对象: {cmap_name}")
        except (AttributeError, KeyError):
            print(f"信息: 在 colorcet.cm 中未直接找到名为 '{cmap_name}' 的对象。")
            try:
                # 尝试从 matplotlib.cm 获取 Colormap 对象
                mpl_cmap = mcm.get_cmap(cmap_name)
                # 从 Colormap 对象中采样颜色列表 (例如 256 个颜色)
                # datashader 通常期望一个颜色列表
                # mpl_cmap() 函数接收 0-1 的值，返回 RGBA 元组
                # 我们需要一个颜色十六进制字符串或RGB元组的列表
                # 一个简单的方法是采样 matplotlib cmap 并转换为 datashader 接受的格式
                # 但 datashader 的 tf.shade 通常也能直接处理 matplotlib 的 Colormap 对象。
                # 不过，为了更保险，我们可以尝试传递一个颜色列表。

                # 简单起见，tf.shade 应该也能直接处理 matplotlib 的 Colormap 对象。
                # 如果不行，再考虑从 mpl_cmap 对象提取颜色列表。
                # 我们先直接尝试传递 mpl_cmap 对象，如果还报错，再修改这里。
                selected_cmap_for_datashader = mpl_cmap  # 直接传递 matplotlib Colormap 对象
                print(f"信息: 找到 Matplotlib 颜色映射对象: {cmap_name}。尝试直接使用。")

            except ValueError:  # matplotlib.cm.get_cmap 在找不到时会抛出 ValueError
                print(f"警告: 在 colorcet 和 Matplotlib 中均未找到名为 '{cmap_name}' 的颜色映射。")
                print(f"将直接使用字符串 '{cmap_name}' 作为 cmap 名称，依赖 tf.shade() 的内置解析（可能失败）。")
                selected_cmap_for_datashader = cmap_name  # 作为最后的尝试，直接传递字符串
            except Exception as e_mpl:  # 其他可能的 matplotlib 错误
                print(f"警告: 尝试从 Matplotlib 获取 cmap '{cmap_name}' 时出错: {e_mpl}")
                print(f"将直接使用字符串 '{cmap_name}' 作为 cmap 名称。")
                selected_cmap_for_datashader = cmap_name
        # --- 结束修改点 ---

        # 使用 selected_cmap_for_datashader
        img = tf.shade(agg_to_shade, cmap=selected_cmap_for_datashader, how='eq_hist')
        img = tf.set_background(img, color=background_color)
        shade_time = time.time(); print(f"着色完成 (用时: {shade_time - shade_start_time:.2f}s)")

        # --- 5. 保存图像 ---
        print(f"正在保存可视化图像到: {output_png}")
        output_dir = os.path.dirname(output_png);
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        pil_image = img.to_pil(); pil_image.save(output_png)
        save_time = time.time(); print(f"图像已成功保存到: {os.path.abspath(output_png)} (用时: {save_time - shade_time:.2f}s)")
        total_time = time.time() - start_time; print(f"--- 可视化完成 --- (总用时: {total_time:.2f}s)")

    except Exception as e:
        print(f"错误: Datashader 可视化或保存时出错: {e}")
        traceback.print_exc()

# --- 命令行接口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Datashader 可视化轨迹点的宏观分布。")
    parser.add_argument('--csv_path', type=str, required=True, help='输入的轨迹点 CSV 文件路径。需要包含 lat, lon 列。')
    parser.add_argument('--output_png', type=str, required=True, help='输出的可视化 PNG 图片文件路径。')
    parser.add_argument('--width', type=int, default=1000, help='输出图像的宽度（像素）。')
    parser.add_argument('--height', type=int, default=1000, help='输出图像的高度（像素）。')
    parser.add_argument('--cmap', type=str, default='fire', # 保持 'fire' 作为默认值
                        help='Datashader 颜色映射名称 (e.g., fire, hot, viridis, bmy)。优先从 colorcet 查找。')
    parser.add_argument('--background', type=str, default='black', help='图像背景色 (e.g., black, white)。')
    parser.add_argument('--no_log_scale', action='store_true', help='禁用对点计数的对数缩放。')
    args = parser.parse_args()

    visualize_datashader(args.csv_path, args.output_png,
                         canvas_width=args.width, canvas_height=args.height,
                         cmap_name=args.cmap, background_color=args.background,
                         log_scale=not args.no_log_scale)