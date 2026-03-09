# single_exp.py
import argparse
import warnings
import time
import numpy as np
import os
import importlib  # To dynamically import predictors
from utils.tools import load_conf, setup_seed
from utils.logger import MultiExpRecorder, ResultLogger  # Keep ResultLogger if needed here? Probably not.
from utils.dataloader import create_dataloaders
from utils.datasplit import split_by_subject, get_subject_list  # Example split strategy


def run_single_exp(dataset_name, preprocess_method, model_name, seed, device, args):
    """Runs a single experiment for a given configuration."""
    print(f"\n--- Running Exp: Data={dataset_name}, Preprocess={preprocess_method}, Model={model_name}, Seed={seed} ---")
    setup_seed(seed)

    # --- Load Configurations ---
    try:
        model_conf = load_conf(method=model_name, dataset=dataset_name, config_type='model')
        dataset_conf = load_conf(dataset=dataset_name, config_type='dataset')
        # Preprocessing config might be needed by predictor/model later, load if necessary
        # preprocess_conf = load_conf(method=preprocess_method, dataset=dataset_name, config_type='preprocessing')
    except FileNotFoundError as e:
        print(f"Error loading configuration: {e}. Skipping experiment.")
        return None  # Return None to indicate failure

    # --- Determine Processed Data Path ---
    processed_data_dir = os.path.join(args.processed_data_path, dataset_name, preprocess_method)
    # print(f"DEBUG_LOAD: Attempting to load data from directory used by Dataloader: {processed_data_dir}")
    if not os.path.exists(processed_data_dir):
        print(f"Error: Processed data not found at {processed_data_dir}. Run preprocessing first.")
        print(f"Expected command: python preprocessing/preprocess_data.py --dataset {dataset_name} --preprocess_method {preprocess_method}")
        return None

    # --- Data Splitting (Example: Cross-Subject) ---
    try:
        all_subjects = get_subject_list(processed_data_dir)
        # Use seed for reproducible splits across runs if desired, but split should be same for same dataset/seed combo
        split_seed = args.split_seed  # Use a fixed seed for splitting subjects
        train_subjects, val_subjects, test_subjects = split_by_subject(
            all_subjects,
            test_size=getattr(dataset_conf.split, 'test_size', 0.2),
            val_size=getattr(dataset_conf.split, 'val_size', 0.1),
            random_state=split_seed
        )
        # Handle case where validation set might be empty
        if not val_subjects:
            print("Warning: No validation subjects found. Using test set for validation during training.")
            # Need to decide how to handle this: Use test set for validation? Skip validation?
            # Option 1: Use test set for validation (potential data leakage for final eval)
            val_subjects = test_subjects
            # Option 2: Skip validation (rely on fixed epochs or train loss) - requires predictor modification
            # return None # Or raise error

    except Exception as e:
        print(f"Error during data splitting: {e}")
        return None

    # --- Create DataLoaders ---
    try:
        batch_size = model_conf.training.batch_size
        train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_dataloaders(
            processed_data_dir, train_subjects, val_subjects, test_subjects, batch_size, args.num_workers
        )

        # +++ 添加数据抽查 +++
        # print("-" * 20 + " DEBUGGING LOADED DATA " + "-" * 20)
        # try:
        #     # 从训练集中取一个样本检查
        #     sample_data_tensor, sample_label = train_ds[0]  # 获取第一个 epoch 的数据
        #     sample_data_np = sample_data_tensor.numpy()  # 转为 numpy
        #     print(f"DEBUG_DATA: Shape of one sample epoch: {sample_data_np.shape}")  # (Channels, TimePoints)
        #     print(f"DEBUG_DATA: Label of this sample: {sample_label}")
        #     # 打印第一个通道前 10 个点的值
        #     print(f"DEBUG_DATA: First 10 points of first channel: {sample_data_np[0, :10]}")
        #     # 打印第一个通道的均值、标准差、最大最小值
        #     print(
        #         f"DEBUG_DATA: Stats of first channel - Mean: {np.mean(sample_data_np[0, :]):.4f}, Std: {np.std(sample_data_np[0, :]):.4f}, Min: {np.min(sample_data_np[0, :]):.4f}, Max: {np.max(sample_data_np[0, :]):.4f}")
        #     # (可选，如果安装了 matplotlib) 绘制第一个通道波形
        #     # import matplotlib.pyplot as plt
        #     # plt.figure()
        #     # plt.plot(sample_data_np[0, :])
        #     # plt.title(f"Sample Epoch (First Channel) - Label: {sample_label}")
        #     # plt.xlabel("Time Points")
        #     # plt.ylabel("Amplitude")
        #     # plt.show() # 显示图像，会暂停程序运行
        # except Exception as data_debug_e:
        #     print(f"DEBUG_DATA: Error during data inspection: {data_debug_e}")
        # print("-" * 60)
        # +++ 抽查结束 +++
    except RuntimeError as e:
        print(f"Error creating dataloaders: {e}")
        return None

    # --- Prepare Dataset Info for Predictor ---
    # Infer info from loaded data or config
    # Example: Get shape from the first item
    sample_data, _ = train_ds[0]
    n_channels = sample_data.shape[-2]  # Assuming shape (..., Channels, Time)
    n_times = sample_data.shape[-1]
    n_classes = getattr(dataset_conf, 'n_classes', len(np.unique(train_ds.labels)))  # Infer or use config
    dataset_info = {
        'n_classes': n_classes,
        'n_channels': n_channels,
        'n_times': n_times,
        # Add other relevant info like sampling frequency if needed
        'sfreq': getattr(dataset_conf, 'sfreq', None)
    }
    print(f"Dataset Info: Classes={n_classes}, Channels={n_channels}, Samples/TimePoints={n_times}")

    # --- Instantiate Predictor ---
    try:
        # Assumes predictor class name is ModelName_Predictor
        predictor_class_name = f"{model_name}_Predictor"
        predictor_module = importlib.import_module(f"predictor.{predictor_class_name}")
        predictor_class = getattr(predictor_module, predictor_class_name)
        predictor = predictor_class(model_conf, dataset_info, device)
    except (ModuleNotFoundError, AttributeError, Exception) as e:
        print(f"Error instantiating predictor '{predictor_class_name}': {e}")
        return None

    # --- Train and Evaluate ---
    try:
        # Train the model, returns results from the best validation epoch
        train_results = predictor.train(train_loader, val_loader)

        # Perform final evaluation on the test set using the best model state
        test_loss, test_acc, test_metrics = predictor.test(test_loader)

        # Combine results
        final_results = train_results.copy()  # Contains train/val acc/loss, best_epoch, time
        final_results['test_loss'] = test_loss
        final_results['test_acc'] = test_acc
        # Add other test metrics
        for key, value in test_metrics.items():
            # Avoid overwriting standard keys if they exist (e.g. 'accuracy')
            if key not in ['accuracy']:  # Keep 'test_acc' as the primary accuracy
                final_results[key] = value  # Add f1_macro, kappa, auc etc.

        return final_results

    except Exception as e:
        print(f"Error during training or testing: {e}")
        import traceback
        traceback.print_exc()  # Print detailed traceback
        return None  # Indicate failure
