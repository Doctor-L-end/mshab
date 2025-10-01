import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import open3d as o3d

def robust_normalize_to_01(data):
    """
    健壮地将数据归一化到 [0,1] 范围
    支持 2D、3D、4D 数组（包括批量数据）
    """
    # 转换为浮点类型
    if isinstance(data, torch.Tensor):
        data = data.float()
    else:
        data = data.astype(np.float32)
    
    # 检查空数据
    if data.size == 0:
        return data
    
    # 计算最小值和最大值
    if isinstance(data, torch.Tensor):
        min_val = data.min()
        max_val = data.max()
    else:
        min_val = np.min(data)
        max_val = np.max(data)
    
    # 处理所有值相同的情况
    if min_val == max_val:
        # 如果所有值都是零，返回0.5（中性值）
        if min_val == 0:
            return torch.full_like(data, 0.5) if isinstance(data, torch.Tensor) else np.full_like(data, 0.5)
        # 否则返回零
        else:
            return torch.zeros_like(data) if isinstance(data, torch.Tensor) else np.zeros_like(data)
    
    # 归一化到 [0,1]
    normalized = (data - min_val) / (max_val - min_val)
    
    # 确保在 [0,1] 范围内
    if isinstance(normalized, torch.Tensor):
        normalized = torch.clamp(normalized, 0, 1)
    else:
        normalized = np.clip(normalized, 0, 1)
    
    return normalized

def visualize_image(image, title=""):
    """
    可视化图像，自动处理多维数组
    支持 2D、3D、4D 数组（包括批量数据）
    """
    # 复制数据避免修改原始数据
    if isinstance(image, np.ndarray):
        image = image.copy()
    elif isinstance(image, torch.Tensor):
        image = image.clone().detach()
    
    # 处理批量数据（4D数组）
    if image.ndim == 4:
        # 只显示批次中的第一张图像
        image = image[0] if isinstance(image, torch.Tensor) else image[0]
        return visualize_image(image, title + " (批次第一张)")
    
    # 应用归一化
    image = robust_normalize_to_01(image)
    
    # 转换为 NumPy 数组
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # 处理不同维度的图像数据
    if image.ndim == 2:  # 灰度图像 (H, W)
        # 无需额外处理
        pass
    elif image.ndim == 3:  # 彩色或单通道图像
        if image.shape[0] in [1, 3]:  # PyTorch 格式 (C, H, W)
            image = np.transpose(image, (1, 2, 0))
        elif image.shape[2] == 1:  # 单通道图像 (H, W, 1)
            image = image[:, :, 0]  # 转换为 (H, W)
    else:
        raise ValueError(f"不支持的图像维度: {image.ndim}")
    
    # 可视化
    plt.figure(figsize=(8, 8))
    
    if image.ndim == 2:  # 灰度图像
        plt.imshow(image, cmap='gray')
    else:  # RGB 图像
        plt.imshow(image)
    
    plt.title(title)
    plt.axis('off')
    # plt.show(block=False)
    SAVE_DIR = "test_vis_data"
    os.makedirs(SAVE_DIR, exist_ok=True)  # 创建文件夹（如果不存在）
    save_path = os.path.join(SAVE_DIR, title.replace(" ", "_") + ".png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def visualize_batch(images, title="", ncols=4):
    """
    可视化整个图像批次
    images: 4D数组 (B, C, H, W) 或 (B, H, W, C)
    """
    # 确保是批量数据
    if images.ndim != 4:
        raise ValueError("输入应为4维数组 (batch, height, width, channels) 或 (batch, channels, height, width)")
    
    # 转换为 NumPy 数组
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    
    # 确定通道位置
    if images.shape[1] in [1, 3]:  # (B, C, H, W)
        # 转换为 (B, H, W, C)
        images = np.transpose(images, (0, 2, 3, 1))
    elif images.shape[3] in [1, 3]:  # (B, H, W, C)
        # 已经是正确格式
        pass
    else:
        raise ValueError("无法确定通道维度")
    
    # 归一化整个批次
    images = robust_normalize_to_01(images)
    
    # 计算网格布局
    batch_size = images.shape[0]
    nrows = (batch_size + ncols - 1) // ncols
    
    # 创建图形
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15/ncols*nrows))
    axes = axes.flatten() if nrows > 1 else [axes]
    
    # 显示每张图像
    for i in range(batch_size):
        ax = axes[i]
        img = images[i]
        
        # 处理单通道图像
        if img.shape[2] == 1:
            img = img[:, :, 0]  # 转换为 (H, W)
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        
        ax.set_title(f"图像 {i+1}")
        ax.axis('off')
    
    # 隐藏多余的子图
    for j in range(batch_size, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def visualize_point_cloud_open3d(point_cloud):
    """
    使用Open3D可视化单一批次的点云
    
    参数:
        point_cloud: 形状为 (N, 6) 的点云数据
        batch_idx: 要可视化的批次索引
    """
    # 确保张量在CPU上
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().cpu().numpy()
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    
    # 设置点云坐标
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    
    # 设置点云颜色（假设RGB值在[0,1]范围内）
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])
    
    # 可视化
    o3d.visualization.draw_geometries([pcd])

def save_pointcloud_to_pcd(point_cloud, filename):
    """
    将点云张量保存为 PCD 文件
    
    参数:
        point_cloud_tensor: 形状为 (N, 6) 的点云张量
        filename: 保存的文件路径（例如: "/path/to/save/pointcloud.pcd"）
        batch_idx: 要保存的批次索引
    """
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().cpu().numpy()
    
    
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    
    # 设置点坐标（前3列）
    points = point_cloud[:, :3].astype(np.float32)
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 设置点颜色（后3列），并确保值范围在 [0, 1] 之间
    colors = point_cloud[:, 3:6].astype(np.float32)
    if np.max(colors) > 1.0:
        colors = np.clip(colors, 0, 255) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 保存为 PCD 文件
    o3d.io.write_point_cloud(filename, pcd)
    print(f"点云已成功保存到: {filename}")

def normalize_point_cloud(point_cloud):
    """
    对形状为(B, T, N, 6)的点云进行归一化
    前三个坐标归一化到[-1, 1]，后三个颜色值归一化到[0, 1]
    支持PyTorch Tensor和NumPy数组输入，输出类型与输入一致
    
    参数:
        point_cloud: 形状为(B, T, N, 6)的张量或数组
        
    返回:
        归一化后的点云，形状与输入相同，类型与输入一致
    """
    # 检查输入类型并保存原始类型和设备信息
    is_tensor = torch.is_tensor(point_cloud)
    device = None
    dtype = None
    
    if is_tensor:
        # 保存原始设备信息和数据类型
        device = point_cloud.device
        dtype = point_cloud.dtype
        # 转换为NumPy数组
        point_cloud_np = point_cloud.cpu().numpy()
    else:
        # 已经是NumPy数组
        point_cloud_np = point_cloud
    
    # 获取形状信息
    B, T, N, _ = point_cloud_np.shape
    
    # 分离坐标和颜色
    coords = point_cloud_np[..., :3]  # 前三个坐标为空间坐标
    colors = point_cloud_np[..., 3:]  # 后三个坐标为颜色信息
    
    # 1. 坐标归一化到[-1, 1]
    # 重塑以便计算每个样本的最小最大值
    reshaped_coords = coords.reshape(B, T * N, 3)
    min_coords = np.min(reshaped_coords, axis=1, keepdims=True)
    max_coords = np.max(reshaped_coords, axis=1, keepdims=True)
    
    # 避免除零错误
    range_coords = max_coords - min_coords
    range_coords[range_coords == 0] = 1
    
    # 归一化到[0, 1]然后映射到[-1, 1]
    coords_normalized = 2 * ((coords - min_coords.reshape(B, 1, 1, 3)) / 
                            range_coords.reshape(B, 1, 1, 3)) - 1
    
    # 2. 颜色归一化到[0, 1]
    # 检查颜色是否已经在[0, 1]范围内
    reshaped_colors = colors.reshape(B, T * N, 3)
    min_colors = np.min(reshaped_colors, axis=1)
    max_colors = np.max(reshaped_colors, axis=1)
    
    # 如果颜色值不在[0, 1]范围内，则进行归一化
    if np.any(min_colors < 0) or np.any(max_colors > 1):
        # 假设颜色值在[0, 255]范围内，需要归一化到[0, 1]
        colors_normalized = colors / 255.0
    else:
        # 颜色值已经在[0, 1]范围内，保持不变
        colors_normalized = colors
    
    # 合并归一化后的坐标和颜色
    normalized_point_cloud_np = np.concatenate([coords_normalized, colors_normalized], axis=-1)
    
    # 转换回原始类型
    if is_tensor:
        # 转换为PyTorch Tensor并放回原始设备
        normalized_point_cloud = torch.from_numpy(normalized_point_cloud_np).to(dtype).to(device)
    else:
        # 保持为NumPy数组
        normalized_point_cloud = normalized_point_cloud_np
    
    return normalized_point_cloud

# 测试函数
def test_multidimensional_visualization():
    """测试多维数组的可视化"""
    # 1. 2D 灰度图像
    gray_2d = np.random.rand(256, 256) * 100 - 50
    visualize_image(gray_2d, "2D 灰度图像")
    
    # 2. 3D PyTorch 格式图像 (C, H, W)
    rgb_3d_torch = torch.randn(3, 256, 256) * 5
    visualize_image(rgb_3d_torch, "3D PyTorch 格式图像")
    
    # 3. 3D NumPy 格式图像 (H, W, C)
    rgb_3d_np = np.random.uniform(-10, 10, size=(256, 256, 3))
    visualize_image(rgb_3d_np, "3D NumPy 格式图像")
    
    # 4. 4D 批量图像 (PyTorch 格式)
    batch_torch = torch.randn(8, 3, 128, 128) * 3
    visualize_batch(batch_torch, "4D PyTorch 批量图像", ncols=4)
    
    # 5. 4D 批量图像 (NumPy 格式)
    batch_np = np.random.uniform(-5, 5, size=(6, 128, 128, 3))
    visualize_batch(batch_np, "4D NumPy 批量图像", ncols=3)
    
    # 6. 单通道批量图像
    batch_gray = torch.randn(4, 1, 128, 128) * 2
    visualize_batch(batch_gray, "单通道批量图像")

# 运行测试
if __name__ == "__main__":
    # test_multidimensional_visualization()
    a = np.random.randint(-100, 100, size=(2, 2, 12, 12, 3))
    a = robust_normalize_to_01(a)
    print(a)
    print(a.shape)