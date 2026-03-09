import numpy as np
import os

# 读取单个512分划的NPZ文件
def read_512npz_file(file_path):
    """
    读取512分划的NPZ文件并返回数据和标签
    
    Args:
        file_path: NPZ文件路径
    
    Returns:
        X: 窗口化的EEG数据，形状为(W, win_size, C)
        y: 窗口的标签，形状为(W,)
        metadata: 元数据字典
    """
    # 加载NPZ文件
    data = np.load(file_path, allow_pickle=True)
    
    # 提取数据和标签
    X = data['X']  # 形状: (窗口数量, 窗口大小, 通道数)
    y = data['y']  # 形状: (窗口数量,)
    
    # 提取元数据
    metadata = {
        'win_size': data['win_size'],
        'hop_size': data['hop_size'],
        'threshold': data['threshold'],
        'channel_names': data['channel_names'],
        'sampling_rate': data['sampling_rate'],
        'seizure_intervals': data['seizure_intervals'],
        'patient_id': data['patient_id']
    }
    
    return X, y, metadata

# 批量读取多个NPZ文件
def read_multiple_npz_files(directory):
    """
    批量读取目录中的所有512分划的NPZ文件
    
    Args:
        directory: 包含NPZ文件的目录
    
    Returns:
        all_X: 所有文件的X数据，形状为(W_total, win_size, C)
        all_y: 所有文件的y标签，形状为(W_total,)
        metadata_list: 每个文件的元数据列表
    """
    all_X = []
    all_y = []
    metadata_list = []
    
    # 遍历目录中的所有NPZ文件
    for filename in os.listdir(directory):
        if filename.endswith('_win512_hop512.npz'):
            file_path = os.path.join(directory, filename)
            print(f"Reading file: {filename}")
            
            # 读取单个文件
            X, y, metadata = read_512npz_file(file_path)
            
            # 添加到总数据中
            all_X.append(X)
            all_y.append(y)
            metadata_list.append(metadata)
    
    # 合并所有数据
    all_X = np.vstack(all_X) if all_X else np.array([])
    all_y = np.concatenate(all_y) if all_y else np.array([])
    
    return all_X, all_y, metadata_list

# 示例用法
if __name__ == "__main__":
    # 单个文件示例
    print("=== 单个文件示例 ===")
    example_file = "data/output_npz/chb01_win512_hop512.npz"
    if os.path.exists(example_file):
        X, y, metadata = read_512npz_file(example_file)
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Window size: {metadata['win_size']}")
        print(f"Hop size: {metadata['hop_size']}")
        print(f"Channels: {len(metadata['channel_names'])}")
        print(f"Sampling rate: {metadata['sampling_rate']} Hz")
        print(f"Patient ID: {metadata['patient_id']}")
        print(f"Positive samples: {np.sum(y)}")
        print(f"Negative samples: {len(y) - np.sum(y)}")
    else:
        print(f"File not found: {example_file}")
    
    print("\n=== 批量文件示例 ===")
    # 批量文件示例
    npz_directory = "data/output_npz"
    if os.path.exists(npz_directory):
        all_X, all_y, metadata_list = read_multiple_npz_files(npz_directory)
        print(f"Total X shape: {all_X.shape}")
        print(f"Total y shape: {all_y.shape}")
        print(f"Total samples: {len(all_y)}")
        print(f"Total positive samples: {np.sum(all_y)}")
        print(f"Total negative samples: {len(all_y) - np.sum(all_y)}")
        print(f"Files read: {len(metadata_list)}")
    else:
        print(f"Directory not found: {npz_directory}")
    
    print("\n=== 数据准备示例 (用于模型训练) ===")
    # 数据准备示例 (用于模型训练)
    if 'all_X' in locals() and 'all_y' in locals() and len(all_X) > 0:
        # 划分训练集和测试集
        from sklearn.model_selection import train_test_split
        
        # 随机划分 80% 训练集，20% 测试集
        X_train, X_test, y_train, y_test = train_test_split(
            all_X, all_y, test_size=0.2, random_state=42, stratify=all_y
        )
        
        print(f"Train set: X={X_train.shape}, y={y_train.shape}")
        print(f"Test set: X={X_test.shape}, y={y_test.shape}")
        print(f"Train positive samples: {np.sum(y_train)}")
        print(f"Test positive samples: {np.sum(y_test)}")
    
    print("\n=== 数据格式说明 ===")
    print("X shape: (num_windows, window_size, num_channels)")
    print("y shape: (num_windows,)")
    print("y values: 0 (non-seizure), 1 (seizure)")
