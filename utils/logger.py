# utils/logger.py
import torch
import time
import pandas as pd
import numpy as np
import os


# --- SingleExpRecorder --- (Keep as is from NoisyGL, useful for early stopping)
class SingleExpRecorder:
    def __init__(self, patience=100, criterion='metric'):  # Use 'metric' or 'loss'
        self.patience = patience
        self.criterion = criterion
        self.best_loss = float('inf')
        self.best_metric = -float('inf')  # Maximize metric
        self.wait = 0

    def add(self, loss_val, metric_val):
        flag = False
        # Determine improvement based on criterion
        if self.criterion == 'loss':
            if loss_val < self.best_loss:
                flag = True
                self.best_loss = loss_val
                # Update best_metric too if tracking both
                if metric_val > self.best_metric:
                    self.best_metric = metric_val
        elif self.criterion == 'metric':
            if metric_val > self.best_metric:
                flag = True
                self.best_metric = metric_val
                # Update best_loss too if tracking both
                if loss_val < self.best_loss:
                    self.best_loss = loss_val
        # Add 'either' or 'both' if needed
        else:  # Default: assume improvement if metric improves OR loss decreases significantly
            if metric_val > self.best_metric or loss_val < self.best_loss * 0.99:  # Small tolerance for loss 加入=？
                flag = True
                self.best_metric = max(self.best_metric, metric_val)
                self.best_loss = min(self.best_loss, loss_val)

        if flag:
            self.wait = 0
        else:
            self.wait += 1

        flag_earlystop = self.patience is not None and self.wait >= self.patience
        return flag, flag_earlystop


# --- MultiExpRecorder --- (Adjust keys in add_result and get_statistics)
class MultiExpRecorder(object):
    def __init__(self, runs):
        # Store results per run. Each run's list will hold dicts or values.
        self.results = [{} for _ in range(runs)]
        self.run_metrics = [  # Define the metrics to record per run
            "train_acc", "valid_acc", "test_acc",
            "train_loss", "valid_loss", "test_loss",
            "f1_macro", "f1_weighted", "kappa", "auc",  # Add relevant test metrics
            "total_time", "best_epoch"
        ]

    def add_result(self, run, result_dict):
        """ Add performance of a new run. result_dict contains metrics."""
        assert 0 <= run < len(self.results)
        for key in self.run_metrics:
            # Use get with default value (e.g., 0.0 or np.nan)
            self.results[run][key] = result_dict.get(key, np.nan)

    def get_statistics(self):
        """ Calculate mean and std deviation across runs for each metric. """
        if not self.results or not self.results[0]:  # Check if any results were added
            print("Warning: No results recorded in MultiExpRecorder.")
            # Return empty stats or NaNs
            stats = {key: {'mean': np.nan, 'std': np.nan} for key in self.run_metrics}
            return stats

        all_stats = {}
        num_runs = len(self.results)

        # Convert list of dicts to dict of lists
        data = {key: [self.results[run].get(key, np.nan) for run in range(num_runs)]
                for key in self.run_metrics}

        print(f'\n--- Statistics Across {num_runs} Runs ---')
        for key in self.run_metrics:
            values = np.array(data[key])
            # Filter out NaNs before calculating stats
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values)
                all_stats[key] = {'mean': mean_val, 'std': std_val}
                # Print key stats (adjust formatting as needed)
                if 'acc' in key or 'f1' in key or 'auc' in key or 'kappa' in key:
                    print(f'{key:<15}: {mean_val * 100:.2f} ± {std_val * 100:.2f}')
                elif 'loss' in key:
                    print(f'{key:<15}: {mean_val:.4f} ± {std_val:.4f}')
                elif 'time' in key:
                    print(f'{key:<15}: {mean_val:.2f}s ± {std_val:.2f}s')
                else:
                    print(f'{key:<15}: {mean_val:.2f} ± {std_val:.2f}')
            else:
                all_stats[key] = {'mean': np.nan, 'std': np.nan}
                print(f'{key:<15}: NaN (No valid runs)')
        print('----------------------------------\n')
        return all_stats


# --- ResultLogger --- (Adapt index naming, column naming, and data extraction)
class ResultLogger(object):
    def __init__(self, method_list, data_list, preprocess_list, runs):
        self.file_name = str(time.strftime("%Y-%m-%d_%H-%M-%S"))
        log_dir = './log/'
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, self.file_name + '.txt')
        self.tex_path = os.path.join(log_dir, self.file_name + '.tex')
        self.excel_path = os.path.join(log_dir, self.file_name + '.xlsx')
        self.runs = runs

        # Create MultiIndex for pandas DataFrames
        index_tuples = []
        for data in data_list:
            for prep in preprocess_list:
                index_tuples.append((data, prep))
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=["Dataset", "Preprocessing"])

        # Main result table (e.g., Test Accuracy)
        self.results_table_test_acc = pd.DataFrame(index=multi_index, columns=method_list)
        # Add other tables for F1, AUC, Time etc. if needed
        self.results_table_test_f1_macro = pd.DataFrame(index=multi_index, columns=method_list)
        self.results_table_time = pd.DataFrame(index=multi_index, columns=method_list)
        # Keep separate mean and std tables for Excel if desired
        self.results_table_test_acc_mean = pd.DataFrame(index=multi_index, columns=method_list)
        self.results_table_test_acc_std = pd.DataFrame(index=multi_index, columns=method_list)

    def dump_record(self, method_name, data_name, preprocess_name, run_statistics):
        """ Dumps the aggregated statistics for a specific setting. """

        # Extract key statistics (mean ± std)
        def get_stat_str(stats_dict, key, multiplier=100, precision=2):
            stat = stats_dict.get(key, {'mean': np.nan, 'std': np.nan})
            mean = stat['mean'] * multiplier
            std = stat['std'] * multiplier
            if np.isnan(mean) or np.isnan(std):
                return "NaN"
            return f"{mean:.{precision}f} ± {std:.{precision}f}"

        def get_stat_mean(stats_dict, key, multiplier=1, precision=2):
            stat = stats_dict.get(key, {'mean': np.nan})
            mean = stat['mean'] * multiplier
            if np.isnan(mean):
                return "NaN"
            return f"{mean:.{precision}f}"

        def get_stat_std(stats_dict, key, multiplier=1, precision=2):
            stat = stats_dict.get(key, {'std': np.nan})
            std = stat['std'] * multiplier
            if np.isnan(std):
                return "NaN"
            return f"{std:.{precision}f}"

        test_acc_str = get_stat_str(run_statistics, 'test_acc')
        test_f1_str = get_stat_str(run_statistics, 'f1_macro')  # Use f1_macro or f1_weighted
        time_str = get_stat_str(run_statistics, 'total_time', multiplier=1)

        # --- Update Pandas DataFrames ---
        loc_tuple = (data_name, preprocess_name)
        self.results_table_test_acc.loc[loc_tuple, method_name] = test_acc_str
        self.results_table_test_f1_macro.loc[loc_tuple, method_name] = test_f1_str
        self.results_table_time.loc[loc_tuple, method_name] = time_str

        # Update mean/std tables for Excel
        self.results_table_test_acc_mean.loc[loc_tuple, method_name] = get_stat_mean(run_statistics, 'test_acc',
                                                                                     multiplier=100)
        self.results_table_test_acc_std.loc[loc_tuple, method_name] = get_stat_std(run_statistics, 'test_acc',
                                                                                   multiplier=100)

        # --- Write to Files ---
        # Plain Text Log
        message = (f"| Dataset: {data_name:<15} | Preprocessing: {preprocess_name:<20} | Method: {method_name:<15} | "
                   f"Test Acc: {test_acc_str:<15} | Test F1m: {test_f1_str:<15} | Time: {time_str:<15} |\n")
        print(message.strip())  # Print to console
        with open(self.log_path, 'a') as f:
            f.write(message)

        # LaTeX Table (Test Accuracy) - Customize formatting as needed
        try:
            with open(self.tex_path, 'w') as f:
                # Create a copy for LaTeX formatting (e.g., replace ± with \pm)
                tex_df = self.results_table_test_acc.copy()
                for col in tex_df.columns:
                    tex_df[col] = tex_df[col].str.replace(' ± ', ' $\\pm$ ', regex=False)
                    # Handle NaN -> '-'
                    tex_df[col] = tex_df[col].replace('NaN', '-', regex=False)
                    # Add $ signs
                    tex_df[col] = '$' + tex_df[col] + '$'
                    tex_df[col] = tex_df[col].replace('$NaN$', '-', regex=False)  # Fix NaN again

                # Format MultiIndex for LaTeX
                tex_df.index.names = ['Dataset', 'Preprocessing']  # Ensure names are set

                latex_string = tex_df.to_latex(
                    na_rep='-', escape=False,  # escape=False needed for \pm
                    bold_rows=True,  # Optional
                    caption=f'Test Accuracy (\%) Across {self.runs} Runs',
                    label='tab:results_acc',  # Optional label
                    multicolumn_format='c'  # Center column headers
                )
                f.write(latex_string)
        except Exception as e:
            print(f"Error writing LaTeX file: {e}")

        # Excel File (Multiple Sheets)
        try:
            with pd.ExcelWriter(self.excel_path, engine='xlsxwriter') as writer:
                self.results_table_test_acc.to_excel(writer, sheet_name='Test Acc (Mean ± Std)', na_rep='NaN')
                self.results_table_test_f1_macro.to_excel(writer, sheet_name='Test F1 Macro (Mean ± Std)', na_rep='NaN')
                self.results_table_time.to_excel(writer, sheet_name='Time (Mean ± Std)', na_rep='NaN')
                # Add mean/std sheets
                self.results_table_test_acc_mean.to_excel(writer, sheet_name='Test Acc (Mean)', na_rep='NaN')
                self.results_table_test_acc_std.to_excel(writer, sheet_name='Test Acc (Std)', na_rep='NaN')
                # Add sheets for other metrics if calculated (F1, AUC, Kappa...)
        except Exception as e:
            print(f"Error writing Excel file: {e}")
