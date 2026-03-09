"""
快速检查数据维度
"""
import numpy as np
import os
import sys

def quick_check(data_dir):
    """快速检查维度一致性"""
    print(f"快速检查: {data_dir}")

    # 遍历所有文件
    shapes = {}
    problems = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                try:
                    data = np.load(file_path, mmap_mode='r')
                    shape = data.shape

                    if shape not in shapes:
                        shapes[shape] = []
                    shapes[shape].append(file_path)

                except Exception as e:
                    print(f"❌ 加载失败: {file_path} - {e}")

    # 显示结果
    print(f"\n📊 发现 {len(shapes)} 种不同的维度：")

    for i, (shape, files) in enumerate(shapes.items()):
        print(f"{i+1}. 维度 {shape}: {len(files)} 个文件")
        if len(files) < 5:  # 如果文件少，显示具体文件名
            for f in files[:3]:
                print(f"   - {os.path.relpath(f, data_dir)}")
        else:
            print(f"   示例: {os.path.relpath(files[0], data_dir)}")
            print(f"         ... 还有 {len(files)-1} 个文件")

    # 如果有多种维度，显示问题
    if len(shapes) > 1:
        print(f"\n⚠️ 警告: 发现维度不一致！")

        # 找出最常见的维度
        most_common = max(shapes.items(), key=lambda x: len(x[1]))
        print(f"最常见的维度: {most_common[0]} ({len(most_common[1])}个文件)")

        # 列出异常文件
        print(f"\n异常文件列表：")
        for shape, files in shapes.items():
            if shape != most_common[0]:
                for file in files[:5]:  # 最多显示5个
                    print(f"  {os.path.relpath(file, data_dir)}: {shape}")
                if len(files) > 5:
                    print(f"  ... 还有 {len(files)-5} 个类似文件")

    return len(shapes) == 1  # 返回是否一致

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法: python quick_check.py <数据目录>")
        sys.exit(1)

    is_consistent = quick_check(sys.argv[1])
    sys.exit(0 if is_consistent else 1)