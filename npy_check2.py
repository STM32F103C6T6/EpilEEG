"""
快速显示.npy文件大小的简单脚本
"""
import os
import numpy as np
import sys

def show_npy_sizes_simple(directory):
    """快速显示.npy文件大小"""
    print(f"📁 目录: {directory}")
    print("=" * 80)

    total_size = 0
    file_count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)

                try:
                    # 文件大小
                    file_size = os.path.getsize(file_path)

                    # 加载数据获取维度
                    data = np.load(file_path, mmap_mode='r')

                    # 计算可读大小
                    size_kb = file_size / 1024
                    size_mb = size_kb / 1024

                    if size_mb >= 1:
                        size_str = f"{size_mb:.2f} MB"
                    elif size_kb >= 1:
                        size_str = f"{size_kb:.2f} KB"
                    else:
                        size_str = f"{file_size} B"

                    # 显示信息
                    rel_path = os.path.relpath(file_path, directory)
                    print(f"{rel_path}")
                    print(f"  ├── 大小: {size_str} ({file_size} 字节)")
                    print(f"  ├── 维度: {data.shape}")
                    print(f"  ├── 通道: {data.shape[0] if len(data.shape) >= 1 else 'N/A'}")
                    print(f"  ├── 时间点: {data.shape[1] if len(data.shape) >= 2 else 'N/A'}")
                    print(f"  └── 数据类型: {data.dtype}")
                    print()

                    total_size += file_size
                    file_count += 1

                except Exception as e:
                    print(f"{os.path.relpath(file_path, directory)} - 错误: {e}")

    # 打印总计
    print("=" * 80)
    print(f"📊 总计:")
    print(f"  文件数: {file_count}")
    print(f"  总大小: {total_size / 1024 / 1024:.2f} MB ({total_size} 字节)")

    if file_count > 0:
        print(f"  平均大小: {total_size / file_count / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法: python show_npy_sizes.py <目录路径>")
        print("示例: python show_npy_sizes.py E:\\zyf\\phytiumhattest\\EpilEEG-master\\processed_data")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.exists(directory):
        print(f"错误: 目录不存在 - {directory}")
        sys.exit(1)

    show_npy_sizes_simple(directory)