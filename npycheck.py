import numpy as np
import os

# --- 用户配置 ---
# 将 'your_file.npy' 替换为你的.npy文件名，或提供完整路径
NPY_FILE_PATH = 'processed_data/epilepsy_eeg/epilepsy_filter_epoch/Patient_004_labels.npy'


def read_and_print_npy(file_path):
    """
    读取指定的.npy文件，并打印其基本信息和内容。
    """
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 '{file_path}' 不存在。")
        return

    try:
        # 2. 使用 np.load() 读取文件
        print(f"--- 正在读取文件: {file_path} ---")
        data = np.load(file_path)

        # 3. 打印数组的基本信息
        print("\n文件加载成功！数组信息如下：")
        print(f"  - 形状 (Shape): {data.shape}")
        print(f"  - 数据类型 (Data Type): {data.dtype}")
        print(f"  - 维度 (Dimensions): {data.ndim}")
        print(f"  - 元素总数 (Size): {data.size}")

        # 4. 打印数组内容
        # 为了防止数组过大导致满屏输出，可以做一个判断
        if data.size > 1000:  # 如果元素超过1000个
            print("\n数组内容 (仅展示部分):")
            # print(data) # NumPy默认会用...省略中间部分，所以直接打印也可以
            print(data[:5, :5] if data.ndim == 2 else data[:10])  # 更精细的控制
        else:
            print("\n数组完整内容:")
            print(data)

    except Exception as e:
        print(f"读取或处理文件时发生错误: {e}")


if __name__ == '__main__':
    read_and_print_npy(NPY_FILE_PATH)