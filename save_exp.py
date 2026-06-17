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
from utils.datasplit import (
    split_by_subject,
    get_subject_list,
    split_by_subject_kfold,
    split_by_subject_kfold_mixed_val,
    split_by_subject_loso_mixed_val,
    get_all_sample_list,
    split_all_samples_kfold_mixed,
)

import argparse
import warnings
import time
import numpy as np
import os
import importlib
import torch  # <--- 如果之前没有，请添加
import torch.onnx # <--- 添加这一行
from utils.tools import load_conf, setup_seed
# ... 其他导入 ...



import numpy as np

def infer_num_classes(dataset_conf, ds):
    if hasattr(dataset_conf, 'n_classes'):
        return dataset_conf.n_classes

    if hasattr(ds, 'labels'):
        labels = ds.labels
    elif hasattr(ds, 'dataset') and hasattr(ds.dataset, 'labels'):
        labels = ds.dataset.labels[ds.indices]
    else:
        raise AttributeError("Cannot infer n_classes from dataset.")

    return len(np.unique(labels))

def save_exp(dataset_name, preprocess_method, model_name, seed, device, args):
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
    # --- Data Splitting: Group K-Fold by Subject ---
    try:
        all_subjects = get_subject_list(processed_data_dir)
        split_seed = args.split_seed
        n_splits = args.n_splits
        fold_idx = args.fold_idx

        if getattr(args, 'all_mixed', False):
            all_samples = get_all_sample_list(processed_data_dir)
            train_subjects, val_subjects, test_subjects = split_all_samples_kfold_mixed(
                all_samples=all_samples,
                n_splits=args.n_splits,
                fold_idx=args.fold_idx,
                val_size=getattr(dataset_conf.split, 'val_size', 0.1),
                random_state=args.split_seed
            )

            print(f"All-mixed sample split done: fold {fold_idx + 1}/{n_splits}")
            print(f"Train mixed samples: {len(train_subjects)}")
            print(f"Val mixed samples:   {len(val_subjects)}")
            print(f"Test mixed samples:  {len(test_subjects)}")

        else:
            all_subjects = get_subject_list(processed_data_dir)

            if getattr(args, 'loso', False):
                train_subjects, val_subjects, test_subjects = split_by_subject_loso_mixed_val(
                    all_subjects=all_subjects,
                    fold_idx=args.fold_idx
                )
                n_splits = len(all_subjects)

            elif getattr(args, 'mixed_val', False):
                train_subjects, val_subjects, test_subjects = split_by_subject_kfold_mixed_val(
                    all_subjects=all_subjects,
                    n_splits=args.n_splits,
                    fold_idx=args.fold_idx,
                    random_state=args.split_seed
                )

            else:
                train_subjects, val_subjects, test_subjects = split_by_subject_kfold(
                    all_subjects=all_subjects,
                    n_splits=args.n_splits,
                    fold_idx=args.fold_idx,
                    val_size=getattr(dataset_conf.split, 'val_size', 0.1),
                    random_state=args.split_seed
                )

            if not val_subjects:
                print("Warning: No validation subjects found. Using test subjects as validation subjects.")
                val_subjects = test_subjects

            print(f"K-Fold split done: fold {fold_idx + 1}/{n_splits}")
            print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
            print(f"Val subjects   ({len(val_subjects)}): {val_subjects}")
            print(f"Test subjects  ({len(test_subjects)}): {test_subjects}")

    except Exception as e:
        print(f"Error during K-Fold subject splitting: {e}")
        return None

    # --- Create DataLoaders ---
    try:
        batch_size = model_conf.training.batch_size
        train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_dataloaders(
            processed_data_dir=processed_data_dir,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
            batch_size=batch_size,
            num_workers=args.num_workers,
            mixed_val=getattr(args, 'mixed_val', False),
            val_ratio=getattr(dataset_conf.split, 'val_size', 0.1),
            split_seed=args.split_seed,
            all_mixed=getattr(args, 'all_mixed', False),
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
    n_classes = infer_num_classes(dataset_conf, train_ds)  # Infer or use config
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

        # /////
        # --- 在 predictor.train(...) 之后 ---

        # ====================================================================
        # +++【最终修正版】导出为纯 CPU ONNX 模型的方案 +++
        # ====================================================================
        print("\n--- Exporting the best model to a CPU-only ONNX format ---")

        # 使用 copy.deepcopy 创建一个模型的完整、独立的副本，避免修改原始模型
        import copy

        # 1. 获取训练好的、仍在GPU上的原始模型
        model_on_gpu = predictor.model
        model_on_gpu.eval()  # 将原始模型也设置为评估模式，以防万一

        # 2. 创建一个深拷贝副本
        model_copy = copy.deepcopy(model_on_gpu)

        # 3. 将副本彻底移动到 CPU
        model_for_export = model_copy.to('cpu')
        print("A deep copy of the model has been created and moved to CPU for export.")

        # 4. 在 CPU 上创建虚拟输入
        dummy_input_cpu = torch.randn(1, n_channels, n_times, device='cpu')
        print(f"Created a dummy input tensor on CPU with shape: {dummy_input_cpu.shape}")

        # 5. 定义 ONNX 文件保存路径
        save_base_path = "experiment_outputs"
        onnx_dir = os.path.join(save_base_path, 'onnx_models_cpu')
        os.makedirs(onnx_dir, exist_ok=True)
        onnx_filename = f"{dataset_name}_{preprocess_method}_{model_name}_seed{seed}_cpu.onnx"
        onnx_path = os.path.join(onnx_dir, onnx_filename)

        try:
            torch.onnx.export(
                model_for_export,  # 使用副本进行导出
                dummy_input_cpu,
                onnx_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                              'output': {0: 'batch_size'}}
            )
            print(f"Successfully exported CPU-only ONNX model to: {onnx_path}")

            # ... (验证部分代码可以保持不变) ...
            # import onnx
            # import onnxruntime
            # providers = ['CPUExecutionProvider']
            # ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
            # ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_cpu.numpy()}
            # ort_outs = ort_session.run(None, ort_inputs)
            # print("ONNX model validation successful using CPUExecutionProvider!")

        except Exception as e:
            print(f"An error occurred during ONNX export or validation: {e}")
            import traceback
            traceback.print_exc()



        # ====================================================================
        # +++ ONNX 导出代码结束，原始 predictor 对象状态未被改变 +++
        # ====================================================================

        # 现在可以安全地在 GPU 上继续执行测试
        #print("Evaluating on Test Set...")
        #test_loss, test_acc, test_metrics = predictor.test(test_loader)
        # /////


        # Train the model, returns results from the best validation epoch
        train_results = predictor.train(train_loader, val_loader,test_loader)

        # /////
        # --- 在 predictor.train(...) 之后 ---

        # ====================================================================
        # +++【最终修正版】导出为纯 CPU ONNX 模型的方案 +++
        # ====================================================================
        print("\n--- Exporting the best model to a CPU-only ONNX format ---")

        # 使用 copy.deepcopy 创建一个模型的完整、独立的副本，避免修改原始模型
        import copy

        # 1. 获取训练好的、仍在GPU上的原始模型
        model_on_gpu = predictor.model
        model_on_gpu.eval()  # 将原始模型也设置为评估模式，以防万一

        # 2. 创建一个深拷贝副本
        model_copy = copy.deepcopy(model_on_gpu)

        # 3. 将副本彻底移动到 CPU
        model_for_export = model_copy.to('cpu')
        print("A deep copy of the model has been created and moved to CPU for export.")

        # 4. 在 CPU 上创建虚拟输入
        dummy_input_cpu = torch.randn(1, n_channels, n_times, device='cpu')
        print(f"Created a dummy input tensor on CPU with shape: {dummy_input_cpu.shape}")

        # 5. 定义 ONNX 文件保存路径
        save_base_path = "experiment_outputs"
        onnx_dir = os.path.join(save_base_path, 'onnx_models_cpu')
        os.makedirs(onnx_dir, exist_ok=True)
        onnx_filename = f"{dataset_name}_{preprocess_method}_{model_name}_seed{seed}_cpu.onnx"
        onnx_path = os.path.join(onnx_dir, onnx_filename)

        # 6. 执行导出（使用副本）
        try:
            torch.onnx.export(
                model_for_export,  # 使用副本进行导出
                dummy_input_cpu,
                onnx_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                              'output': {0: 'batch_size'}}
            )
            print(f"Successfully exported CPU-only ONNX model to: {onnx_path}")

            # ... (验证部分代码可以保持不变) ...
            # import onnx
            # import onnxruntime
            # providers = ['CPUExecutionProvider']
            # ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
            # ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_cpu.numpy()}
            # ort_outs = ort_session.run(None, ort_inputs)
            # print("ONNX model validation successful using CPUExecutionProvider!")

        except Exception as e:
            print(f"An error occurred during ONNX export or validation: {e}")
            import traceback
            traceback.print_exc()

        # ====================================================================
        # +++ ONNX 导出代码结束，原始 predictor 对象状态未被改变 +++
        # ====================================================================

        # 现在可以安全地在 GPU 上继续执行测试
        print("Evaluating on Test Set...")
        #test_loss, test_acc, test_metrics = predictor.test(test_loader)
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
