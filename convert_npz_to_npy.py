import numpy as np
import os
import glob

# 转换单个512分划的NPZ文件为项目预期的npy格式
def convert_npz_to_npy(npz_file, output_dir):
    """
    将512分划的NPZ文件转换为项目预期的npy格式
    
    Args:
        npz_file: NPZ文件路径
        output_dir: 输出目录
    """
    # 加载NPZ文件
    data = np.load(npz_file, allow_pickle=True)
    
    # 提取数据和标签
    X = data['X']  # 形状: (窗口数量, 窗口大小, 通道数)
    y = data['y']  # 形状: (窗口数量,)
    
    # 提取患者ID
    patient_id = data['patient_id']
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为npy文件
    epochs_file = os.path.join(output_dir, f"{patient_id}_epochs.npy")
    labels_file = os.path.join(output_dir, f"{patient_id}_labels.npy")
    
    # 转换数据维度顺序: (W, win_size, C) -> (W, C, win_size)
    # 因为项目中的dataloader期望形状为 (n_epochs, n_channels, n_times)
    X_transposed = np.transpose(X, (0, 2, 1))
    
    np.save(epochs_file, X_transposed)
    np.save(labels_file, y)
    
    print(f"Converted {npz_file} to:")
    print(f"  Epochs: {epochs_file} (shape: {X_transposed.shape})")
    print(f"  Labels: {labels_file} (shape: {y.shape})")
    
    return patient_id

# 批量转换多个NPZ文件
def batch_convert_npz_to_npy(npz_directory, output_base_dir, preprocess_name="chbmit_512win"):
    """
    批量转换目录中的所有512分划的NPZ文件
    
    Args:
        npz_directory: 包含NPZ文件的目录
        output_base_dir: 输出基础目录
        preprocess_name: 预处理方法名称
    """
    # 构建输出目录路径
    output_dir = os.path.join(output_base_dir, "epilepsy_eeg", preprocess_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有512分划的NPZ文件
    npz_files = glob.glob(os.path.join(npz_directory, "*_win512_hop512.npz"))
    
    print(f"Found {len(npz_files)} NPZ files to convert")
    
    # 转换每个文件
    converted_patients = []
    for npz_file in npz_files:
        try:
            patient_id = convert_npz_to_npy(npz_file, output_dir)
            converted_patients.append(patient_id)
        except Exception as e:
            print(f"Error converting {npz_file}: {e}")
    
    # 创建dataset_info.json文件
    create_dataset_info(output_dir, converted_patients)
    
    print(f"\nConversion completed. Converted {len(converted_patients)} patients")
    print(f"Output directory: {output_dir}")
    
    return output_dir

# 创建dataset_info.json文件
def create_dataset_info(output_dir, patient_ids):
    """
    创建dataset_info.json文件，包含数据集信息
    
    Args:
        output_dir: 输出目录
        patient_ids: 转换的患者ID列表
    """
    import json
    
    # 加载一个文件来获取元数据
    sample_file = glob.glob(os.path.join(output_dir, "*_epochs.npy"))[0]
    sample_data = np.load(sample_file)
    
    n_channels = sample_data.shape[1]
    n_times = sample_data.shape[2]
    n_classes = 2  # 癫痫分类通常是二分类
    
    dataset_info = {
        "n_channels": n_channels,
        "n_times": n_times,
        "n_classes": n_classes,
        "subjects": sorted(patient_ids),
        "sampling_rate": 256.0,  # CHB-MIT数据集的采样率
        "preprocessing": "chbmit_512win",
        "description": "CHB-MIT dataset preprocessed into 512-sample windows"
    }
    
    info_file = os.path.join(output_dir, "dataset_info.json")
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Created dataset_info.json: {info_file}")

# 示例用法
if __name__ == "__main__":
    # 输入和输出目录
    npz_directory = "data/output_npz"
    output_base_dir = "processed_data"
    
    # 批量转换
    output_dir = batch_convert_npz_to_npy(npz_directory, output_base_dir)
    
    print("\n=== 转换完成 ===")
    print(f"转换后的数据可以在以下目录找到:")
    print(output_dir)
    print("\n接下来，您可以使用以下命令运行训练:")
    print("python total_exp.py --datasets epilepsy_eeg --preprocess_methods chbmit_512win --methods HAT MedFormer --runs 5 --split_seed 666 --start_seed 666")
