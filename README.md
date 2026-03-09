# EEG Benchmark 项目

这是一个用于脑电图 (EEG) 分类任务的基准测试 (Benchmark) 项目框架。它旨在提供一个模块化、可扩展且可复现的环境，方便研究人员添加新的数据集、预处理方法和深度学习模型，并进行公平的比较。

## 项目特点

*   **模块化设计:** 将数据处理、模型定义、训练评估逻辑和日志记录分离，易于理解和扩展。
*   **数据处理流程:** 支持从原始数据 (EDF, GDF, NPY 等) 到标准化预处理数据 (NPY) 的转换。
*   **模型集成:** 方便添加新的 EEG 分类模型 (如 EEGNet, DeepConvNet, 以及您自定义的 HAT, MedFormer 等)。
*   **配置驱动:** 通过 YAML 文件管理数据集、预处理和模型的参数，方便调整和记录实验设置。
*   **结果记录:** 自动记录每次运行的详细指标，并生成汇总统计结果 (TXT, XLSX, LaTeX)。
*   **可复现性:** 支持设置随机种子，确保实验结果在相同配置下可复现。
*   **受试者独立评估:** 支持按受试者划分数据集，进行可靠的泛化能力评估。

## 项目结构

```
EEG_Benchmark/
├── config/ # 配置文件目录
│ ├── _dataset/ # 数据集特定配置 (sfreq, event_id, split...)
│ ├── _preprocessing/ # 预处理流程配置 (filter, epoching...)
│ └── <model_name>/ # 模型特定配置 (hyperparameters, training settings...)
├── data/ # 存放原始 EEG 数据集 (按数据集名称分子目录)
│ └── <dataset_name>/
│ └── Patient_XXX/ # 按受试者存放原始文件 (例如 .edf)
├── processed_data/ # 存放预处理后的标准化数据 (NPY 格式)
│ └── <dataset_name>/
│ └── <preprocess_method_name>/ # 按预处理方法分子目录
│ ├── Patient_XXX_epochs.npy
│ ├── Patient_XXX_labels.npy
│ └── dataset_info.json # 预处理元数据
├── models/ # 模型定义目录
│ ├── init.py
│ ├── layers/ # (可选) 存放模型共享的基础层
│ │ ├── init.py
│ │ └── ...
│ ├── hat.py # HAT 模型实现
│ ├── medformer.py # MedFormer 模型实现
│ └── ... # 其他模型文件
├── predictor/ # 训练和评估逻辑封装 (Predictor 类)
│ ├── init.py
│ ├── Base_Predictor.py # Predictor 基类
│ ├── HAT_Predictor.py
│ ├── MedFormer_Predictor.py
│ └── ... # 其他模型的 Predictor
├── preprocessing/ # 数据预处理模块
│ ├── init.py
│ ├── methods/ # 具体的预处理方法实现
│ │ ├── init.py
│ │ ├── base_preprocessor.py # Preprocessor 基类
│ │ ├── edf_preprocessor.py # 处理 EDF 格式的示例
│ │ └── ...
│ └── preprocess_data.py # 执行预处理的主脚本
├── utils/ # 辅助工具模块
│ ├── init.py
│ ├── dataloader.py # 加载预处理后数据的 DataLoader
│ ├── datasplit.py # 数据集划分逻辑 (按受试者等)
│ ├── logger.py # 实验结果记录
│ ├── masking.py # (可选) 注意力掩码等工具
│ ├── metrics.py # 评估指标计算
│ └── tools.py # 通用工具 (加载配置, 设置种子等)
├── log/ # 保存实验结果日志文件
├── .gitignore
├── LICENSE # 项目许可证 (暂未开源)
├── README.md # 本文档
├── requirements.txt # 项目依赖库
├── single_exp.py # (内部调用) 运行单次实验的脚本
└── total_exp.py # 运行完整 Benchmark 实验的主脚本
```

## 环境设置

1.  **克隆项目:**
    ```bash
    git clone <your-repository-url>
    cd EpilepsyEEG
    ```
2.  **创建虚拟环境 (推荐):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```
    核心依赖包括: `torch`, `numpy`, `scipy`, `pandas`, `mne`, `scikit-learn`, `ruamel.yaml`, `xlsxwriter`, `einops` 。请根据实际情况更新 `requirements.txt`。

## 使用流程

### 1. 准备数据

*   将您的原始 EEG 数据集按照项目结构要求放入 `data/<your_dataset_name>/Patient_XXX/` 目录下。确保文件名包含区分状态的信息（如果需要基于文件名打标签）。

### 2. 配置数据集

*   在 `config/_dataset/` 目录下为您的数据集创建一个 YAML 配置文件 (例如 `my_dataset.yaml`)。
*   在文件中定义 `name`, `loader_type` (例如 `'edf'`, `'gdf'`, `'epilepsy_edf'`), `subjects` 列表, `sfreq`, 以及（如果使用基于事件的分割）`event_id` 和 `label_map`，还有数据划分策略 `split`。

### 3. 配置并运行预处理

*   在 `config/_preprocessing/` 目录下创建或修改一个预处理配置文件 (例如 `my_preprocess_config.yaml`)。
*   定义预处理步骤和参数（例如是否滤波 `filter`, 分段方式 `epoching` - 固定长度或基于事件，是否重采样 `resample` 等）。
*   根据您数据的原始格式，在 `preprocessing/methods/` 下创建或使用合适的 Preprocessor 类 (继承自 `base_preprocessor.py`)。确保 `preprocess_data.py` 中的 `PREPROCESSOR_MAP` 包含您的 `loader_type` 到 Preprocessor 类的映射。
*   运行预处理脚本：
    ```bash
    python -m preprocessing.preprocess_data --dataset <your_dataset_name> --preprocess_config_name <your_preprocess_config>
    ```
    例如:
    ```bash
    python -m preprocessing.preprocess_data --dataset epilepsy_eeg --preprocess_config_name epilepsy_filter_epoch
    ```
*   预处理后的 `.npy` 文件和 `dataset_info.json` 将保存在 `processed_data/<your_dataset_name>/<your_preprocess_config>/` 目录下。

### 4. 添加模型 (如果需要)

*   将模型定义代码放入 `models/` 目录下 (例如 `models/my_model.py`)。如果模型使用了自定义的基础层，建议将这些层放在 `layers/` 子目录下。
*   修改模型的 `__init__` 方法，使其接收 `model_conf` (模型特定配置) 和 `dataset_info` (包含 n\_channels, n\_times, n\_classes 等) 作为参数。
*   在 `predictor/` 目录下创建一个对应的 Predictor 类 (例如 `MyModel_Predictor.py`)，继承自 `Base_Predictor`。
    *   实现 `_build_model` 方法，在其中实例化您的模型，注意正确传递 `self.model_conf.model` 和 `self.dataset_info`。
    *   如果模型需要特殊的训练或评估逻辑（例如不同的 `forward` 参数），重写 `_train_epoch` 或 `evaluate` 方法。

### 5. 配置模型训练

*   在 `config/` 目录下为您的模型创建一个子目录 (例如 `config/my_model/`)。
*   在该子目录下为每个 (数据集, 预处理方法) 组合创建一个模型配置文件 (例如 `my_model_epilepsy_eeg.yaml`)。
*   配置文件结构应包含 `model:` (模型超参数) 和 `training:` (训练参数，如 optimizer, lr, epochs, batch_size, scheduler 等) 两部分。

### 6. 运行 Benchmark 实验

*   使用 `total_exp.py` 脚本运行完整的 Benchmark 流程。通过命令行参数指定要测试的数据集、预处理方法、模型、运行次数等。
    ```bash
    python total_exp.py \
        --datasets <dataset1> <dataset2> ... \
        --preprocess_methods <preprocess1> <preprocess2> ... \
        --methods <model1> <model2> ... \
        --runs <number_of_runs> \
        --split_seed <seed_for_subject_split> \
        --start_seed <seed_for_first_run> \
        --device <cuda:0_or_cpu> \
        --processed_data_path ./processed_data/
    ```
    例如:
    ```bash
    python total_exp.py --datasets epilepsy_eeg --preprocess_methods epilepsy_filter_epoch epilepsy_nofilter_epoch --methods HAT MedFormer --runs 5 --split_seed 666 --start_seed 666
    ```

### 7. 查看结果

*   训练过程中的指标会打印到控制台。
*   每个模型/设置组合运行完成后，会打印平均性能和标准差。
*   详细的日志文件（包括每次运行的指标和最终汇总表）会保存在 `log/` 目录下，以时间戳命名。

## 贡献

欢迎通过 Pull Request 或 Issue 对本项目进行贡献。可能的贡献方向包括：

*   添加对新数据集的支持。
*   实现新的 EEG 预处理方法。
*   集成新的 EEG 分类模型。
*   改进评估指标或日志记录功能。
*   优化代码结构和效率。
*   添加交叉验证功能。

## 许可证

本项目暂未开源。