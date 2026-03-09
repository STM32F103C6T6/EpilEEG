# preprocessing/preprocess_data.py
import argparse
import os
import importlib
import mne
import numpy as np
import json
from utils.tools import load_conf, setup_seed

# Mapping from dataset config 'loader_type' to preprocessor class
# Add new entries here when you create new preprocessor subclasses
PREPROCESSOR_MAP = {
    'edf': 'EDFPreprocessor',
    'gdf': 'GDFPreprocessor',
    'npy': 'NPYPreprocessor',
    'epilepsy_edf': 'EpilepsyEDFPreprocessor',
    'depression_npy': 'DepressionNPYPreprocessor',
    'chbmit': 'CHBMITPreprocessor',
    # Add 'mat', 'bdf', etc. as needed
}


def get_preprocessor_class(loader_type_name):
    """Dynamically imports and returns the preprocessor class."""
    if loader_type_name not in PREPROCESSOR_MAP:
        raise ValueError(f"Unsupported loader_type: '{loader_type_name}'. "
                         f"Available types: {list(PREPROCESSOR_MAP.keys())}")

    class_name = PREPROCESSOR_MAP[loader_type_name]
    module_name = f".methods.{loader_type_name}_preprocessor"  # Assumes file naming convention

    try:
        # Use relative import from within the preprocessing package
        preprocess_module = importlib.import_module(module_name, package='preprocessing')
        preprocessor_class = getattr(preprocess_module, class_name)
        return preprocessor_class
    except (ModuleNotFoundError, AttributeError, ImportError) as e:
        print(f"Error loading preprocessor class '{class_name}' from '{module_name}': {e}")
        raise  # Re-raise the exception


def main(args):
    setup_seed(args.seed)  # Seed for any random steps in preprocessing
    print("test")

    # --- Load Configurations ---
    try:
        dataset_conf = load_conf(dataset=args.dataset, config_type='dataset')
        # Preprocessing config name should match the file in config/_preprocessing/
        # preprocess_conf = load_conf(path=os.path.join('./config/_preprocessing/', args.preprocess_config_name + '.yaml'))
        preprocess_config_path = os.path.join('./config/_preprocessing/', args.preprocess_config_name + '.yaml')
        # print(f"DEBUG: Attempting to load preprocess config from: {preprocess_config_path}")  # 确认路径
        if not os.path.exists(preprocess_config_path):
            print(f"DEBUG: *** ERROR: Config file NOT FOUND at specified path! ***")
            return  # Exit if file doesn't exist

        preprocess_conf = load_conf(path=preprocess_config_path)
        if preprocess_conf is None:
            print(f"DEBUG: *** ERROR: load_conf returned None! Check load_conf function. ***")
            return
        # Store the name back into the config object for reference
        preprocess_conf.name = args.preprocess_config_name

        # --- 在这里添加详细的打印语句 ---
        # print("-" * 20 + " DEBUGGING PREPROCESS_CONF " + "-" * 20)
        # print(f"DEBUG: Loaded preprocess_conf object: {preprocess_conf}")
        # print(f"DEBUG: Type of preprocess_conf: {type(preprocess_conf)}")
        #
        # if hasattr(preprocess_conf, 'epoching'):
        #     print(f"DEBUG: preprocess_conf.epoching object: {preprocess_conf.epoching}")
        #     print(f"DEBUG: Type of preprocess_conf.epoching: {type(preprocess_conf.epoching)}")
        #     # Check attributes using getattr AND direct access if it's a Namespace
        #     duration_found_getattr = getattr(preprocess_conf.epoching, 'epoch_duration', '!!! NOT FOUND via getattr !!!')
        #     print(f"DEBUG: getattr(preprocess_conf.epoching, 'epoch_duration', ...): {duration_found_getattr}")
        #     # Try accessing directly if it's a Namespace
        #     try:
        #         duration_direct = preprocess_conf.epoching.epoch_duration
        #         print(f"DEBUG: preprocess_conf.epoching.epoch_duration (direct access): {duration_direct}")
        #     except AttributeError:
        #         print("DEBUG: *** AttributeError on direct access to preprocess_conf.epoching.epoch_duration ***")
        #
        #     # Also check for overlap
        #     overlap_found_getattr = getattr(preprocess_conf.epoching, 'overlap', '!!! NOT FOUND via getattr !!!')
        #     print(f"DEBUG: getattr(preprocess_conf.epoching, 'overlap', ...): {overlap_found_getattr}")
        #
        # else:
        #     print("DEBUG: *** 'epoching' attribute NOT FOUND in loaded preprocess_conf ***")
        # print("-" * 60)
        # --- 打印语句结束 ---
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading configuration: {e}")
        return

    # --- Determine Paths ---
    raw_data_dir = os.path.join(args.raw_data_path, args.dataset)
    # Output directory now includes the preprocessing config name for clarity
    processed_data_dir = os.path.join(args.processed_data_path, args.dataset, args.preprocess_config_name)
    os.makedirs(processed_data_dir, exist_ok=True)

    print(f"Starting preprocessing for dataset: {args.dataset}")
    print(f"Using preprocessing configuration: {args.preprocess_config_name}")

    # --- Instantiate Correct Preprocessor ---
    try:
        loader_type = getattr(dataset_conf, 'loader_type', None)
        if not loader_type:
            # Attempt to infer loader type from common file extensions if not specified
            first_file = next((f for f in os.listdir(raw_data_dir) if not f.startswith('.')), None)  # Find any file
            if first_file:
                ext = os.path.splitext(first_file)[1].lower()
                if ext == '.edf':
                    loader_type = 'edf'
                elif ext == '.gdf':
                    loader_type = 'gdf'
                elif ext == '.npy':
                    loader_type = 'npy'
                # Add more inferences
                else:
                    print(f"Warning: Could not infer loader_type from file extension '{ext}'.")
            if not loader_type:
                raise ValueError(f"'loader_type' not specified in dataset config for '{args.dataset}' and could not be inferred.")
            print(f"Inferred loader_type: '{loader_type}'")

        PreprocessorClass = get_preprocessor_class(loader_type)
        preprocessor = PreprocessorClass(
            dataset_conf=dataset_conf,
            preprocess_conf=preprocess_conf,
            raw_data_dir=raw_data_dir,
            processed_data_dir=processed_data_dir,
            force_rerun=args.force_rerun
        )
    except (ValueError, ImportError, Exception) as e:
        print(f"Error setting up preprocessor: {e}")
        return

    if hasattr(preprocessor, 'process_all_files') and callable(preprocessor.process_all_files):
        print("Using process_all_files method...")
        processed_ids, failed_ids = preprocessor.process_all_files()
    else:
        # 回退到按 subject 处理的逻辑 (适用于之前的 Preprocessor)
        print("Using process_subject method...")
        subject_ids = getattr(dataset_conf, 'subjects', [])
        if isinstance(subject_ids, range): subject_ids = list(subject_ids)
        if not subject_ids:
            print("Error: No subjects defined in dataset config.")
            return

        processed_ids = []
        failed_ids = []
        for subject_id in subject_ids:
            success = preprocessor.process_subject(subject_id)
            if success:
                processed_ids.append(subject_id)
            else:
                failed_ids.append(subject_id)

    print("\n--- Preprocessing Summary ---")
    print(f"Successfully processed data associated with {len(processed_ids)} original subject IDs.")
    if failed_ids:
        print(f"Failed to process data associated with {len(failed_ids)} original subject IDs: {failed_ids}")

    print("Preprocessing finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Data Preprocessing")
    parser.add_argument('--raw_data_path', type=str, default='./data/', help='Path to parent directory containing raw dataset folders')
    parser.add_argument('--processed_data_path', type=str, default='./processed_data/', help='Path to save processed data')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset folder (must match config file name)')
    parser.add_argument('--preprocess_config_name', type=str, required=True, help='Name of the preprocessing config file in config/_preprocessing/ (without .yaml extension)')
    parser.add_argument('--seed', type=int, default=666, help='Random seed for any stochastic preprocessing steps')
    parser.add_argument('--force_rerun', action='store_true', help='Force reprocessing even if output files exist')
    args = parser.parse_args()
    main(args)

# Example Usage:
# python -m preprocessing.preprocess_data --dataset physionet_eegmmidb --preprocess_config_name basic_filter_epoch
# python -m preprocessing.preprocess_data --dataset bciciv_2a --preprocess_config_name basic_filter_epoch
# python -m preprocessing.preprocess_data --dataset my_npy_dataset --preprocess_config_name basic_filter_epoch --force_rerun
