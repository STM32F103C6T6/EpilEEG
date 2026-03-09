# utils/tools.py
import torch
import numpy as np
import random
import argparse
import os
import ruamel.yaml as yaml
import warnings
from ruamel.yaml import YAML
from types import SimpleNamespace

warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)


def dict_to_sns(d):
    """Recursively converts a dictionary to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_sns(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_sns(item) for item in d]
    else:
        return d


def load_conf(path: str = None, method: str = None, dataset: str = None, config_type: str = "model"):
    """
    Function to load config file.
    config_type can be 'model', 'dataset', or 'preprocessing'.
    """
    if path is None:
        if method is None or dataset is None:
            # For dataset or preprocessing config, method might not be needed
            if config_type == 'model' and method is None:
                raise ValueError("Method name is required for loading model config.")
            if dataset is None:
                raise ValueError("Dataset name is required for loading config.")

        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

        if config_type == "model":
            path = os.path.join(base_dir, method, f"{method}_{dataset}.yaml")
        elif config_type == "dataset":
            path = os.path.join(base_dir, "_dataset", f"{dataset}.yaml")
        elif config_type == "preprocessing":
            # Assumes preprocessing config name matches method name in preprocessing/methods
            # Or uses a specific naming convention like method_config.yaml
            path = os.path.join(base_dir, "_preprocessing", f"{method}_config.yaml")  # Example convention
        else:
            raise ValueError(f"Unknown config_type: {config_type}")

        if not os.path.exists(path):
            # Try a default config if specific one doesn't exist (optional)
            # default_path = os.path.join(base_dir, ..., "default.yaml")
            # if os.path.exists(default_path): path = default_path else:
            raise FileNotFoundError(f"Config file not found at {path}")

    yaml_loader = YAML(typ='safe')
    try:
        with open(path, "r", encoding='utf-8') as f:
            conf_dict = yaml_loader.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {path}")
    except UnicodeDecodeError as ude:  # 更具体地捕获编码错误
        raise UnicodeDecodeError(f"Encoding error reading {path}: {ude}. Ensure the file is saved as UTF-8.")
    except Exception as e:
        raise Exception(f"Error loading or parsing YAML file {path}: {e}")

    if not isinstance(conf_dict, dict):
        raise TypeError(f"Configuration file {path} did not load as a dictionary.")
    conf = dict_to_sns(conf_dict)
    return conf


def save_conf(path: str, conf: argparse.Namespace):
    """Function to save config file."""
    yaml_saver = YAML()
    yaml_saver.default_flow_style = False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml_saver.dump(vars(conf), f)
    print(f'Config file saved to {path}')


def setup_seed(seed):
    """
    Setup random seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Potentially set environment variable for further determinism
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # or ':16:8'
    print(f"Random seed set to {seed}")

# Add other utility functions if needed, e.g., for EEG specific tasks
