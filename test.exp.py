# test_exp.py
"""
测试专用程序 - 加载已训练模型进行评估
支持单个模型测试、批量模型测试、多折测试
"""

import argparse
import warnings
import time
import numpy as np
import os
import json
import torch
from pathlib import Path
import sys
import importlib
from tabulate import tabulate
from datetime import datetime
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.tools import setup_seed
from utils.dataloader import create_dataloaders
from utils.datasplit import get_subject_list  # 获取受试者列表
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, cohen_kappa_score


def load_model_checkpoint(model_name, checkpoint_path, device='cuda:0'):
    """
    加载已训练的模型
    """
    print(f"Loading model checkpoint from: {checkpoint_path}")

    # 动态导入模型类
    try:
        if model_name == 'HAT':
            from models.hat_model import HATModel
            model_class = HATModel
        elif model_name == 'MedFormer':
            from models.medformer_model import MedFormer
            model_class = MedFormer
        elif model_name == 'EEGNet':
            from models.eegnet import EEGNet
            model_class = EEGNet
        else:
            # 尝试从predictor模块导入
            predictor_class_name = f"{model_name}_Predictor"
            predictor_module = importlib.import_module(f"predictor.{predictor_class_name}")
            predictor_class = getattr(predictor_module, predictor_class_name)
            # 获取模型实例
            # 注意：这里需要先创建predictor，然后获取其model
            print(f"Note: For {model_name}, loading through predictor interface")
            return None, None  # 需要特殊处理
    except ImportError as e:
        print(f"Error importing model {model_name}: {e}")
        return None, None

    # 加载检查点
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None, None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None

    # 解析配置信息
    model_config = checkpoint.get('config', {})

    # 创建模型实例
    try:
        model = model_class(**model_config)

        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            # 尝试直接加载
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        print(f"✓ Successfully loaded {model_name} model")
        return model, checkpoint

    except Exception as e:
        print(f"Error instantiating model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def evaluate_model_on_test_set(model, test_loader, device='cuda:0'):
    """
    在测试集上评估模型
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_probabilities = []

    inference_times = []

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data = data.to(device)
            targets = targets.to(device)

            # 记录推理时间
            start_time = time.time()

            # 前向传播
            outputs = model(data)

            # 计算推理时间
            batch_time = time.time() - start_time
            inference_times.append(batch_time)

            # 获取预测结果
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(test_loader)} "
                      f"Avg time: {np.mean(inference_times) * 1000:.2f}ms")

    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)

    # 计算评估指标
    metrics = {}

    # 基础指标
    metrics['accuracy'] = accuracy_score(all_targets, all_preds)
    metrics['precision_weighted'] = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    # 宏平均
    metrics['precision_macro'] = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    # Cohen's Kappa
    metrics['kappa'] = cohen_kappa_score(all_targets, all_preds)

    # 混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(all_targets, all_preds)

    # 推理时间统计
    metrics['avg_inference_time_ms'] = np.mean(inference_times) * 1000
    metrics['std_inference_time_ms'] = np.std(inference_times) * 1000
    metrics['total_inference_time_s'] = np.sum(inference_times)

    # 样本统计
    metrics['n_samples'] = len(all_targets)
    metrics['n_correct'] = np.sum(all_predictions == all_targets)

    # 类别详细报告
    class_report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
    metrics['class_report'] = class_report

    # 添加每个类别的精度
    for class_idx in sorted(class_report.keys()):
        if class_idx.isdigit():  # 跳过平均指标
            metrics[f'precision_class_{class_idx}'] = class_report[class_idx]['precision']
            metrics[f'recall_class_{class_idx}'] = class_report[class_idx]['recall']
            metrics[f'f1_class_{class_idx}'] = class_report[class_idx]['f1-score']
            metrics[f'support_class_{class_idx}'] = class_report[class_idx]['support']

    return metrics


def test_single_model(model_name, dataset_name, preprocess_method,
                      checkpoint_path, test_subjects, args, fold_idx=None):
    """
    测试单个模型在特定配置下
    """
    print(f"\n{'=' * 60}")
    print(f"Testing Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Preprocess: {preprocess_method}")
    if fold_idx is not None:
        print(f"Fold: {fold_idx}")
    print(f"Test Subjects: {test_subjects}")
    print(f"{'=' * 60}")

    # 设置随机种子
    setup_seed(args.seed if hasattr(args, 'seed') else 42)

    # 1. 创建数据加载器
    processed_data_dir = os.path.join(args.processed_data_path, dataset_name, preprocess_method)

    if not os.path.exists(processed_data_dir):
        print(f"Error: Processed data not found at {processed_data_dir}")
        return None

    # 创建测试数据加载器
    # 注意：训练和验证集为空
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_dataloaders(
        processed_data_dir=processed_data_dir,
        train_subjects=[],  # 空训练集
        val_subjects=[],  # 空验证集
        test_subjects=test_subjects,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        mixed_val=False,
        all_mixed=False
    )

    print(f"Test dataset size: {len(test_ds)} samples")

    # 2. 加载模型
    model, checkpoint = load_model_checkpoint(model_name, checkpoint_path, args.device)
    if model is None:
        print(f"Failed to load model from {checkpoint_path}")
        return None

    # 3. 评估模型
    print("\nEvaluating on test set...")
    start_time = time.time()
    metrics = evaluate_model_on_test_set(model, test_loader, args.device)
    total_time = time.time() - start_time

    metrics['total_evaluation_time_s'] = total_time
    metrics['model_name'] = model_name
    metrics['dataset_name'] = dataset_name
    metrics['preprocess_method'] = preprocess_method
    metrics['fold_idx'] = fold_idx if fold_idx is not None else -1
    metrics['test_subjects'] = test_subjects
    metrics['checkpoint_path'] = checkpoint_path

    # 添加检查点中的训练信息（如果有）
    if checkpoint:
        if 'best_val_acc' in checkpoint:
            metrics['best_val_acc'] = checkpoint['best_val_acc']
        if 'best_epoch' in checkpoint:
            metrics['best_epoch'] = checkpoint['best_epoch']
        if 'train_loss' in checkpoint:
            metrics['train_loss'] = checkpoint['train_loss']

    return metrics


def find_checkpoints(models_dir, pattern=None):
    """
    在模型目录中查找检查点文件
    """
    checkpoints = {}
    models_dir = Path(models_dir)

    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return checkpoints

    # 查找所有.pth文件
    pth_files = list(models_dir.glob("**/*.pth")) + list(models_dir.glob("**/*.pt"))

    for pth_file in pth_files:
        # 解析文件名获取信息
        filename = pth_file.stem

        # 尝试从文件名解析模型、数据集、预处理方法
        parts = filename.split('_')

        if pattern:
            # 使用模式匹配
            match = True
            for key, value in pattern.items():
                if key == 'model':
                    if value not in filename.lower():
                        match = False
                        break
                elif key == 'dataset':
                    if value not in filename.lower():
                        match = False
                        break
                elif key == 'preprocess':
                    if value not in filename.lower():
                        match = False
                        break
                elif key == 'fold':
                    if f"fold{value}" not in filename.lower() and f"fold_{value}" not in filename.lower():
                        match = False
                        break

            if match:
                checkpoints[str(pth_file)] = {
                    'path': str(pth_file),
                    'model': pattern.get('model'),
                    'dataset': pattern.get('dataset'),
                    'preprocess': pattern.get('preprocess'),
                    'fold': pattern.get('fold')
                }
        else:
            # 没有模式，收集所有文件
            checkpoints[str(pth_file)] = {
                'path': str(pth_file),
                'filename': filename
            }

    return checkpoints


def save_results_to_file(results, output_dir='./test_results'):
    """
    保存测试结果到文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存为CSV
    csv_file = os.path.join(output_dir, f'test_results_{timestamp}.csv')

    # 整理结果
    rows = []
    for result in results:
        if result is None:
            continue

        row = {
            'timestamp': timestamp,
            'model': result['model_name'],
            'dataset': result['dataset_name'],
            'preprocess': result['preprocess_method'],
            'fold': result['fold_idx'],
            'accuracy': result['accuracy'],
            'f1_weighted': result['f1_weighted'],
            'f1_macro': result['f1_macro'],
            'kappa': result['kappa'],
            'n_samples': result['n_samples'],
            'n_correct': result['n_correct'],
            'avg_inference_time_ms': result['avg_inference_time_ms'],
            'test_subjects': str(result['test_subjects']),
            'checkpoint': result['checkpoint_path']
        }

        # 添加每个类别的指标
        for key, value in result.items():
            if key.startswith('f1_class_'):
                class_idx = key.replace('f1_class_', '')
                row[f'f1_class_{class_idx}'] = value
            elif key.startswith('precision_class_'):
                class_idx = key.replace('precision_class_', '')
                row[f'precision_class_{class_idx}'] = value

        rows.append(row)

    # 保存到CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"\nResults saved to: {csv_file}")

    # 保存详细报告
    report_file = os.path.join(output_dir, f'detailed_report_{timestamp}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL TESTING REPORT\n")
        f.write("=" * 80 + "\n\n")

        for i, result in enumerate(results):
            if result is None:
                continue

            f.write(f"\n{'=' * 60}\n")
            f.write(f"Test #{i + 1}: {result['model_name']} on {result['dataset_name']}\n")
            f.write(f"{'=' * 60}\n\n")

            f.write(f"Configuration:\n")
            f.write(f"  Model: {result['model_name']}\n")
            f.write(f"  Dataset: {result['dataset_name']}\n")
            f.write(f"  Preprocess: {result['preprocess_method']}\n")
            f.write(f"  Fold: {result['fold_idx']}\n")
            f.write(f"  Checkpoint: {result['checkpoint_path']}\n")
            f.write(f"  Test Subjects: {result['test_subjects']}\n\n")

            f.write(f"Results:\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"  Precision (weighted): {result['precision_weighted']:.4f}\n")
            f.write(f"  Recall (weighted): {result['recall_weighted']:.4f}\n")
            f.write(f"  F1 (weighted): {result['f1_weighted']:.4f}\n")
            f.write(f"  F1 (macro): {result['f1_macro']:.4f}\n")
            f.write(f"  Cohen's Kappa: {result['kappa']:.4f}\n")
            f.write(f"  Correct/Total: {result['n_correct']}/{result['n_samples']}\n\n")

            f.write(f"Inference Performance:\n")
            f.write(f"  Avg Inference Time: {result['avg_inference_time_ms']:.2f} ms\n")
            f.write(f"  Total Evaluation Time: {result['total_evaluation_time_s']:.2f} s\n\n")

            if 'best_val_acc' in result:
                f.write(f"Training Info (from checkpoint):\n")
                f.write(f"  Best Validation Acc: {result['best_val_acc']:.4f}\n")
                f.write(f"  Best Epoch: {result.get('best_epoch', 'N/A')}\n")
                f.write(f"  Train Loss: {result.get('train_loss', 'N/A')}\n\n")

            f.write(f"Confusion Matrix:\n")
            cm = result['confusion_matrix']
            f.write(f"{cm}\n\n")

            f.write(f"Per-class Metrics:\n")
            for class_idx in sorted([k for k in result.keys() if k.startswith('f1_class_')]):
                class_num = class_idx.replace('f1_class_', '')
                f.write(f"  Class {class_num}: ")
                f.write(f"Precision={result.get(f'precision_class_{class_num}', 0):.4f}, ")
                f.write(f"Recall={result.get(f'recall_class_{class_num}', 0):.4f}, ")
                f.write(f"F1={result.get(f'f1_class_{class_num}', 0):.4f}, ")
                f.write(f"Support={result.get(f'support_class_{class_num}', 0)}\n")

    print(f"Detailed report saved to: {report_file}")

    return csv_file, report_file


def main():
    parser = argparse.ArgumentParser(description="Test trained EEG models")

    # 模型相关参数
    parser.add_argument('--model', type=str, default='HAT', help='Model name to test')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to specific checkpoint file. If not provided, will search in models_dir')
    parser.add_argument('--models_dir', type=str, default='./trained_models',
                        help='Directory containing trained model checkpoints')

    # 数据相关参数
    parser.add_argument('--dataset', type=str, default='epilepsy_eeg', help='Dataset name')
    parser.add_argument('--preprocess_method', type=str, default='basic_filter_epoch',
                        help='Preprocessing method')
    parser.add_argument('--processed_data_path', type=str, default='./processed_data',
                        help='Path containing processed data folders')

    # 测试集配置
    parser.add_argument('--test_subjects', type=str, nargs='+', default=None,
                        help='List of test subject IDs. If not provided, uses all subjects')
    parser.add_argument('--test_split_file', type=str, default=None,
                        help='File containing test subject IDs (one per line)')
    parser.add_argument('--n_test_subjects', type=int, default=5,
                        help='Number of subjects to use for testing (random selection)')

    # 多折测试
    parser.add_argument('--test_all_folds', action='store_true',
                        help='Test all fold checkpoints found in models_dir')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='Specific fold index to test')

    # 批量测试
    parser.add_argument('--batch_test', action='store_true',
                        help='Test multiple models from models_dir')
    parser.add_argument('--model_pattern', type=str, default=None,
                        help='Pattern to match model files (e.g., "HAT_epilepsy")')

    # 评估参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Directory to save results')

    args = parser.parse_args()

    print("=" * 80)
    print("TRAINED MODEL TESTING TOOL")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)

    # 设置随机种子
    setup_seed(args.seed)

    # 确定测试集受试者
    if args.test_split_file and os.path.exists(args.test_split_file):
        # 从文件读取
        with open(args.test_split_file, 'r') as f:
            test_subjects = [int(line.strip()) for line in f if line.strip()]
        print(f"Loaded {len(test_subjects)} test subjects from {args.test_split_file}")
    elif args.test_subjects:
        # 从命令行参数读取
        test_subjects = [int(subj) for subj in args.test_subjects]
        print(f"Using {len(test_subjects)} test subjects from command line")
    else:
        # 从数据目录获取所有受试者
        processed_data_dir = os.path.join(args.processed_data_path, args.dataset, args.preprocess_method)
        if os.path.exists(processed_data_dir):
            all_subjects = get_subject_list(processed_data_dir)
            # 随机选择一部分作为测试集
            np.random.seed(args.seed)
            if len(all_subjects) > args.n_test_subjects:
                test_subjects = np.random.choice(all_subjects, args.n_test_subjects, replace=False).tolist()
            else:
                test_subjects = all_subjects
            print(f"Randomly selected {len(test_subjects)} test subjects from {len(all_subjects)} total subjects")
        else:
            print(f"Error: Processed data directory not found: {processed_data_dir}")
            return

    test_subjects = sorted(test_subjects)
    print(f"Test subjects: {test_subjects}")

    results = []

    if args.test_all_folds:
        # 测试所有折
        print(f"\nTesting all folds for {args.model} on {args.dataset}...")

        # 查找所有折的检查点
        fold_results = []
        for fold_idx in range(5):  # 假设最多5折
            checkpoint_pattern = {
                'model': args.model.lower(),
                'dataset': args.dataset.lower(),
                'preprocess': args.preprocess_method.lower(),
                'fold': fold_idx
            }

            checkpoints = find_checkpoints(args.models_dir, checkpoint_pattern)

            if not checkpoints:
                print(f"No checkpoint found for fold {fold_idx}")
                continue

            # 使用找到的第一个检查点
            checkpoint_path = list(checkpoints.values())[0]['path']

            print(f"\nTesting fold {fold_idx} with checkpoint: {checkpoint_path}")

            # 测试该折
            fold_metrics = test_single_model(
                model_name=args.model,
                dataset_name=args.dataset,
                preprocess_method=args.preprocess_method,
                checkpoint_path=checkpoint_path,
                test_subjects=test_subjects,
                args=args,
                fold_idx=fold_idx
            )

            if fold_metrics:
                fold_results.append(fold_metrics)

        # 计算平均结果
        if fold_results:
            print("\n" + "=" * 80)
            print("CROSS-VALIDATION RESULTS SUMMARY")
            print("=" * 80)

            # 计算各项指标的平均值和标准差
            metrics_to_average = ['accuracy', 'f1_weighted', 'f1_macro', 'kappa']
            summary = {}

            for metric in metrics_to_average:
                values = [r[metric] for r in fold_results]
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)

            # 打印结果
            table_data = []
            for i, result in enumerate(fold_results):
                table_data.append([
                    i,
                    result['accuracy'],
                    result['f1_weighted'],
                    result['f1_macro'],
                    result['kappa'],
                    result['n_samples']
                ])

            # 添加平均值行
            table_data.append([
                'Mean ± Std',
                f"{summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}",
                f"{summary['f1_weighted_mean']:.4f} ± {summary['f1_weighted_std']:.4f}",
                f"{summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}",
                f"{summary['kappa_mean']:.4f} ± {summary['kappa_std']:.4f}",
                ''
            ])

            print(tabulate(table_data,
                           headers=['Fold', 'Accuracy', 'F1 Weighted', 'F1 Macro', 'Kappa', 'Samples'],
                           tablefmt='grid'))

            results.extend(fold_results)

    elif args.batch_test:
        # 批量测试多个模型
        print(f"\nBatch testing models from {args.models_dir}...")

        # 查找匹配的检查点
        checkpoints = find_checkpoints(args.models_dir)

        if not checkpoints:
            print(f"No checkpoints found in {args.models_dir}")
            return

        print(f"Found {len(checkpoints)} checkpoints")

        for i, (checkpoint_path, info) in enumerate(checkpoints.items()):
            print(f"\n[{i + 1}/{len(checkpoints)}] Testing: {info.get('filename', checkpoint_path)}")

            # 从文件名解析模型名（简单实现，可根据实际情况调整）
            filename = info.get('filename', '')
            model_name = args.model  # 默认使用命令行参数

            # 尝试从文件名推断
            if 'hat' in filename.lower():
                model_name = 'HAT'
            elif 'medformer' in filename.lower() or 'med' in filename.lower():
                model_name = 'MedFormer'
            elif 'eegnet' in filename.lower():
                model_name = 'EEGNet'

            # 测试该模型
            metrics = test_single_model(
                model_name=model_name,
                dataset_name=args.dataset,
                preprocess_method=args.preprocess_method,
                checkpoint_path=checkpoint_path,
                test_subjects=test_subjects,
                args=args,
                fold_idx=-1  # 未知折
            )

            if metrics:
                results.append(metrics)

    else:
        # 测试单个模型
        if args.checkpoint_path:
            # 使用指定的检查点
            checkpoint_path = args.checkpoint_path
        else:
            # 自动查找检查点
            checkpoint_pattern = {
                'model': args.model.lower(),
                'dataset': args.dataset.lower(),
                'preprocess': args.preprocess_method.lower(),
                'fold': args.fold_idx
            }

            checkpoints = find_checkpoints(args.models_dir, checkpoint_pattern)

            if not checkpoints:
                print(f"No checkpoint found for pattern: {checkpoint_pattern}")
                # 尝试查找任何匹配的检查点
                checkpoints = find_checkpoints(args.models_dir, {
                    'model': args.model.lower(),
                    'dataset': args.dataset.lower()
                })

                if not checkpoints:
                    print(f"No checkpoint found for {args.model} on {args.dataset}")
                    return

            # 使用找到的第一个检查点
            checkpoint_path = list(checkpoints.values())[0]['path']

        print(f"\nTesting single model with checkpoint: {checkpoint_path}")

        metrics = test_single_model(
            model_name=args.model,
            dataset_name=args.dataset,
            preprocess_method=args.preprocess_method,
            checkpoint_path=checkpoint_path,
            test_subjects=test_subjects,
            args=args,
            fold_idx=args.fold_idx
        )

        if metrics:
            results.append(metrics)

    # 保存结果
    if results:
        save_results_to_file(results, args.output_dir)

        # 打印汇总表格
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)

        summary_table = []
        for i, result in enumerate(results):
            summary_table.append([
                i + 1,
                result['model_name'],
                result.get('fold_idx', 'N/A'),
                f"{result['accuracy']:.4f}",
                f"{result['f1_weighted']:.4f}",
                f"{result['f1_macro']:.4f}",
                f"{result['kappa']:.4f}",
                f"{result['n_correct']}/{result['n_samples']}",
                f"{result['avg_inference_time_ms']:.1f}ms"
            ])

        print(tabulate(summary_table,
                       headers=['#', 'Model', 'Fold', 'Accuracy', 'F1(w)', 'F1(m)', 'Kappa', 'Correct/Total',
                                'Infer Time'],
                       tablefmt='grid'))
    else:
        print("\nNo results to save.")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()