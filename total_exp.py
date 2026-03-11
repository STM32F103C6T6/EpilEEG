# total_exp.py
import argparse
import warnings
import os
import copy

import numpy as np
import torch

from utils.tools import setup_seed
from utils.logger import MultiExpRecorder, ResultLogger
from save_exp import save_exp


def average_fold_results(fold_results_list):
    """
    对多个 fold 的结果做平均聚合。
    输入: [dict, dict, ...]
    输出: 一个聚合后的 dict
    """
    if len(fold_results_list) == 0:
        return None

    # 找出所有可能的 metric key
    all_keys = set()
    for res in fold_results_list:
        if res is not None:
            all_keys.update(res.keys())

    aggregated = {}
    for key in all_keys:
        values = []
        for res in fold_results_list:
            if res is None:
                continue
            val = res.get(key, np.nan)
            if isinstance(val, (int, float, np.integer, np.floating)):
                values.append(float(val))

        if len(values) > 0:
            aggregated[key] = float(np.nanmean(values))
        else:
            aggregated[key] = np.nan

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Run EEG Classification Benchmark")
    parser.add_argument('--runs', type=int, default=5, help="Number of experiments (runs) for each setting")
    parser.add_argument('--start_seed', type=int, default=666, help="Starting random seed for runs")
    parser.add_argument('--split_seed', type=int, default=666, help="Fixed random seed for subject splits")
    parser.add_argument('--methods', type=str, nargs='+', default=['HAT', 'MedFormer'],
                        help='Select models/methods to run')
    parser.add_argument('--datasets', type=str, nargs='+', default=['epilepsy_eeg'],
                        help='Select datasets to run')
    parser.add_argument('--preprocess_methods', type=str, nargs='+', default=['basic_filter_epoch'],
                        help='Select preprocessing methods used')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda:0 or cpu)')
    parser.add_argument("--num_workers", type=int, default=0, help="data loader num workers")
    parser.add_argument('--processed_data_path', type=str, default='./processed_data/',
                        help='Path containing processed data folders')

    # K-Fold 参数
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds for subject-wise K-Fold')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='Unused in total_exp main loop; kept for compatibility')
    parser.add_argument('--mixed_val', action='store_true',default=False,
                        help='Use subject-wise KFold for test, but mix train/val subjects and split val within training data')
    parser.add_argument('--loso', action='store_true',default=False, help='Use leave-one-subject-out split')

    args = parser.parse_args()

    print("--- Experiment Configuration ---")
    print(args)
    print("-------------------------------")

    warnings.filterwarnings("ignore", category=UserWarning)

    result_recorder = ResultLogger(args.methods, args.datasets, args.preprocess_methods, args.runs)

    # --- Main Experiment Loop ---
    for data_name in args.datasets:
        for preprocess_method in args.preprocess_methods:
            for model_name in args.methods:

                print(f"\n===== Starting Benchmark for: Data={data_name}, Preprocess={preprocess_method}, Model={model_name} =====")
                run_logger = MultiExpRecorder(runs=args.runs)

                for i in range(args.runs):
                    run_seed = args.start_seed + i
                    print(f"\n--- Run {i + 1}/{args.runs} (Seed: {run_seed}) ---")

                    # 固定本 run 的训练随机性
                    setup_seed(run_seed)

                    fold_results_list = []

                    # ===== 遍历所有折 =====
                    for fold_idx in range(args.n_splits):
                        print(f"\n------ Fold {fold_idx + 1}/{args.n_splits} ------")

                        # 不直接污染原 args，复制一份更安全
                        fold_args = copy.deepcopy(args)
                        fold_args.fold_idx = fold_idx

                        fold_results = save_exp(
                            dataset_name=data_name,
                            preprocess_method=preprocess_method,
                            model_name=model_name,
                            seed=run_seed,
                            device=args.device,
                            args=fold_args
                        )

                        if fold_results is not None:
                            fold_results['fold_idx'] = fold_idx
                            fold_results_list.append(fold_results)
                        else:
                            print(f"Fold {fold_idx + 1} failed. Recording as NaN.")

                    # ===== 对当前 run 的所有 fold 求平均 =====
                    run_results = average_fold_results(fold_results_list)

                    if run_results is not None:
                        print(f"\n--- Run {i + 1} K-Fold Averaged Results ---")
                        for k, v in run_results.items():
                            if isinstance(v, (int, float, np.integer, np.floating)):
                                print(f"{k}: {v:.4f}")
                            else:
                                print(f"{k}: {v}")

                        run_logger.add_result(i, run_results)
                    else:
                        print(f"Run {i + 1} failed on all folds. Recording as NaN.")
                        failed_results = {key: np.nan for key in run_logger.run_metrics}
                        run_logger.add_result(i, failed_results)

                # 多个 run 的统计
                aggregated_stats = run_logger.get_statistics()

                result_recorder.dump_record(
                    method_name=model_name,
                    data_name=data_name,
                    preprocess_name=preprocess_method,
                    run_statistics=aggregated_stats
                )

                print(f"===== Finished Benchmark for: Data={data_name}, Preprocess={preprocess_method}, Model={model_name} =====")

    print("\n--- Benchmark Run Complete ---")
    print(f"Results saved in log directory: ./log/{result_recorder.file_name}*")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU.")
    main()
