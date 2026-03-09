# total_exp.py
import argparse
import warnings
import os

import numpy as np

from utils.tools import setup_seed
from utils.logger import MultiExpRecorder, ResultLogger
from single_exp import run_single_exp  # Import the runner function
from save_exp import save_exp  # Import the runner function

def main():
    parser = argparse.ArgumentParser(description="Run EEG Classification Benchmark")
    parser.add_argument('--runs', type=int, default=5, help="Number of experiments (runs) for each setting")
    parser.add_argument('--start_seed', type=int, default=666, help="Starting random seed for runs")
    parser.add_argument('--split_seed', type=int, default=666, help="Fixed random seed for train/val/test subject splits")
    parser.add_argument('--methods', type=str, nargs='+', default=['HAT', 'MedFormer'],  # Add your model names
                        help='Select models/methods to run')
    parser.add_argument('--datasets', type=str, nargs='+', default=['epilepsy_eeg'],  # Add dataset names
                        help='Select datasets to run')
    parser.add_argument('--preprocess_methods', type=str, nargs='+', default=['basic_filter_epoch'],
                        help='Select preprocessing methods used (must match processed data folders)')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device (cuda:0 or cpu)')
    parser.add_argument("--num_workers", type=int, default=0, help="data loader num workers")
    parser.add_argument('--processed_data_path', type=str, default='./processed_data/', help='Path containing processed data folders')
    # Add any other global arguments needed by run_single_exp if necessary

    args = parser.parse_args()
    print("--- Experiment Configuration ---")
    print(args)
    print("-------------------------------")

    warnings.filterwarnings("ignore", category=UserWarning)  # Suppress common warnings

    # Initialize the main result logger
    result_recorder = ResultLogger(args.methods, args.datasets, args.preprocess_methods, args.runs)

    # --- Main Experiment Loop ---
    for data_name in args.datasets:
        for preprocess_method in args.preprocess_methods:
            for model_name in args.methods:

                print(f"\n===== Starting Benchmark for: Data={data_name}, Preprocess={preprocess_method}, Model={model_name} =====")
                # Recorder for multiple runs of this specific setting
                run_logger = MultiExpRecorder(runs=args.runs)

                for i in range(args.runs):
                    run_seed = args.start_seed + i
                    print(f"\n--- Run {i + 1}/{args.runs} (Seed: {run_seed}) ---")

                    # Ensure seed is set for this specific run's initialization/training randomness
                    setup_seed(run_seed)

                    # Run the single experiment 改
                    run_results = save_exp(
                        dataset_name=data_name,
                        preprocess_method=preprocess_method,
                        model_name=model_name,
                        seed=run_seed,
                        device=args.device,
                        args=args  # Pass all args if needed by single_exp
                    )

                    if run_results is not None:
                        run_logger.add_result(i, run_results)
                    else:
                        print(f"Run {i + 1} failed. Recording as NaN.")
                        # Add NaN results or handle failure appropriately
                        failed_results = {key: np.nan for key in run_logger.run_metrics}
                        run_logger.add_result(i, failed_results)

                # Aggregate results across runs for this setting
                aggregated_stats = run_logger.get_statistics()

                # Dump the aggregated results to the main logger
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
    import torch  # Check device availability early

    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU.")
    main()

# Example Usage:
# python total_exp.py --datasets physionet_eegmmidb bciciv_2a --methods EEGNet ShallowConvNet --preprocess_methods basic_filter_epoch --runs 3
