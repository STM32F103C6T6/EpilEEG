# preprocessing/methods/depression_npy_preprocessor.py
import mne
import os
import glob
import numpy as np
import pandas as pd
import json
import re  # 用于解析文件名
import logging
from ..base_preprocessor import BasePreprocessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # 设置日志级别


class DepressionNPYPreprocessor(BasePreprocessor):
    """
    Preprocessor for depression dataset stored as individual .npy files per subject/recording.
    Performs fixed-length windowing (segmentation) during preprocessing and saves
    each segment as a separate file to mimic the existing benchmark structure.
    Filename format expected: ('normal'|'patient') + subject_id + ... + '.npy'
    """

    def _parse_subject_label(self, filename):
        """从文件名提取被试ID和标签(0=normal, 1=patient)。"""
        # --- 复用您提供的解析逻辑 ---
        if filename.startswith("normal"):
            label = 0
            prefix = "normal"
        elif filename.startswith("patient"):
            label = 1
            prefix = "patient"
        else:
            raise ValueError(f"无法从文件名中识别 normal/patient: {filename}")

        # 改进正则，考虑更多情况
        # 尝试匹配 prefix + id + _MEG
        match_meg = re.match(rf"{prefix}(.*?)_MEG", filename, re.IGNORECASE)  # 忽略大小写
        if match_meg:
            subject_id = match_meg.group(1).strip('_')
        else:
            # 尝试匹配 prefix + id + .npy (或其他结束符)
            match_simple = re.match(rf"{prefix}(.*?)(_|\.npy)", filename, re.IGNORECASE)
            if match_simple:
                subject_id = match_simple.group(1).strip('_')
            else:
                raise ValueError(f"文件名不符合预期格式 (无法提取 subject_id): {filename}")

        if not subject_id:
            subject_id = f"unknown_{filename.split('.')[0]}"  # 生成唯一 ID
            logger.warning(f"无法从 {filename} 提取有效的 subject_id，已生成: '{subject_id}'。")

        return subject_id, label
        # --- 解析逻辑结束 ---

    def process_subject(self, subject_id):
        """
        处理与指定 subject_id 相关的所有 .npy 文件。
        注意：这里的 subject_id 是我们从文件名解析出来的标识符。
        我们需要先找到所有文件，然后按 subject_id 分组处理。
        这个方法在当前场景下不太适用，我们将覆盖主处理逻辑。
        """
        # 这个基类的方法不适用，因为我们不是按文件夹找文件，而是先找所有文件再分组
        pass  # 留空或引发 NotImplementedError

    def _load_raw_data(self, subject_id):
        """
        Loads the raw NPY data for files associated with a specific subject ID.
        This method might not be directly called if process_all_files is used,
        but needs to be implemented to satisfy the abstract base class.
        We'll implement it to load files for a given subject, though the main
        logic in process_all_files doesn't rely on it this way.
        """
        # 这是一个示例实现，加载与给定 subject_id 相关的文件
        # 注意：process_all_files 已经加载了所有文件，这里可能冗余
        # 但为了满足抽象方法要求，我们提供一个实现

        logger.debug(f"[_load_raw_data called for subject: {subject_id}] - This might be redundant if process_all_files is primary.")
        npy_list = []
        # 假设我们能找到所有属于这个 subject_id 的文件路径 (需要修改 _find_subject_files 或类似逻辑)
        # 简化：暂时返回空列表，因为主要逻辑在 process_all_files
        # 如果确实需要在这里加载，需要一个方法来获取属于该 subject_id 的文件路径列表
        # related_files = self._find_files_for_single_subject(subject_id) # 需要实现这个方法
        # for fpath in related_files:
        #     try:
        #         data = np.load(fpath)
        #         if data.ndim == 3 and data.shape[0] == 1: data = data.squeeze(0)
        #         npy_list.append(data) # 或者包装成需要的结构
        #     except Exception as e:
        #         logger.warning(f"Error loading file {fpath} in _load_raw_data: {e}")
        return []  # 返回空列表，因为实际加载在 process_all_files

    def process_all_files(self):
        """
        覆盖基类的处理流程，改为处理目录下所有 npy 文件，
        进行分段，并按解析出的 subject_id 保存。
        """
        logger.info(f"开始处理目录 {self.raw_data_dir} 下的所有 .npy 文件...")
        search_pattern = os.path.join(self.raw_data_dir, "*.npy")
        all_npy_files = sorted(glob.glob(search_pattern))

        if not all_npy_files:
            logger.error(f"在 {self.raw_data_dir} 中未找到任何 .npy 文件。")
            return [], []  # 返回空的已处理和失败列表

        logger.info(f"找到 {len(all_npy_files)} 个 .npy 文件。")

        # 按解析出的 subject_id 分组文件
        subject_files_map = {}
        skipped_parse_count = 0
        for fpath in all_npy_files:
            fname = os.path.basename(fpath)
            try:
                subject_id, label = self._parse_subject_label(fname)
                if subject_id not in subject_files_map:
                    subject_files_map[subject_id] = {"label": label, "files": []}
                elif subject_files_map[subject_id]["label"] != label:
                    logger.warning(f"被试 {subject_id} 的文件标签不一致！ 文件 {fname} (标签 {label}) 与之前的标签 {subject_files_map[subject_id]['label']} 不同。将跳过此文件。")
                    skipped_parse_count += 1
                    continue  # 跳过标签不一致的文件

                subject_files_map[subject_id]["files"].append(fpath)
            except ValueError as e:
                logger.warning(f"解析文件名 {fname} 时出错: {e}。已跳过此文件。")
                skipped_parse_count += 1

        if skipped_parse_count > 0:
            logger.warning(f"共有 {skipped_parse_count} 个文件因解析错误或标签不一致而被跳过。")
        if not subject_files_map:
            logger.error("未能成功解析任何被试信息，无法进行处理。")
            return [], list(all_npy_files)  # 返回空成功列表，所有文件失败

        processed_subjects = []
        failed_subjects = []
        segment_counter = 0  # 全局片段计数器，用于生成唯一文件名
        all_segment_metadata = []  # 收集所有片段的元数据

        # --- 获取分段参数 ---
        epoching_conf = getattr(self.preprocess_conf, 'epoching', None)
        if not epoching_conf or getattr(epoching_conf, 'epoch_duration', None) is None:
            logger.error("预处理配置中缺少 'epoching.epoch_duration' 参数，无法进行分段。")
            # 将所有解析出的 subject 视为失败
            return [], list(subject_files_map.keys())

        segment_length_sec = epoching_conf.epoch_duration
        overlap_ratio = getattr(epoching_conf, 'overlap', 0.0)
        logger.info(f"将使用分段长度 {segment_length_sec} 秒，重叠比例 {overlap_ratio * 100:.0f}%。")

        # 遍历每个被试及其文件
        for subject_id, info in subject_files_map.items():
            logger.info(f"\n处理被试: {subject_id} (标签: {info['label']})")
            subject_processed_successfully = True
            subject_segment_count = 0

            # 创建该被试在 processed_data 下的目录 (如果尚不存在)
            # 注意：我们的目标是保存 segment 文件，而不是病人文件夹
            # os.makedirs(os.path.join(self.processed_data_dir, subject_id), exist_ok=True)

            for fpath in info["files"]:
                fname = os.path.basename(fpath)
                logger.info(f"  处理文件: {fname}")
                try:
                    data = np.load(fpath)  # 加载 NPY 数据
                    if data.ndim == 3 and data.shape[0] == 1:
                        data = data.squeeze(0)  # 变为 (C, L)
                    elif data.ndim != 2:
                        raise ValueError(f"NPY 文件数据维度不是 2 或 (1, C, L): {data.shape}")

                    # --- 获取数据的实际采样率 (非常重要!) ---
                    # 理想情况下，这个信息应该与 NPY 文件一起保存，或在 dataset_conf 中提供
                    # 这里我们先从 dataset_conf 获取，如果不存在则报错
                    sfreq = getattr(self.dataset_conf, 'sfreq', None)
                    if sfreq is None:
                        raise ValueError(f"无法确定文件 {fname} 的采样率。请在 dataset config 中设置 'sfreq'。")

                    segment_length_points = int(segment_length_sec * sfreq)
                    step = int(segment_length_points * (1 - overlap_ratio))
                    if step <= 0: step = 1

                    # --- 可选：在分段前进行标准化 ---
                    # if getattr(self.preprocess_conf, 'normalize', False):
                    #     mean = np.mean(data, axis=1, keepdims=True)
                    #     std = np.std(data, axis=1, keepdims=True)
                    #     std[std == 0] = 1 # 防止除以零
                    #     data = (data - mean) / std
                    #     logger.info("    已应用 Z-score 标准化。")

                    # 滑窗分段
                    num_segments_in_file = 0
                    for i in range(0, data.shape[1] - segment_length_points + 1, step):
                        segment = data[:, i:i + segment_length_points].astype(np.float32)  # 确保类型

                        # 为每个 segment 生成唯一的文件名
                        segment_counter += 1
                        segment_filename_base = f"{subject_id}_seg{segment_counter:06d}"  # 例如 Patient_001_seg000001
                        segment_epoch_file = os.path.join(self.processed_data_dir, f"{segment_filename_base}_epochs.npy")
                        segment_label_file = os.path.join(self.processed_data_dir, f"{segment_filename_base}_labels.npy")

                        # --- 跳过已存在的 Segment (如果 force_rerun=False) ---
                        if not self.force_rerun and os.path.exists(segment_epoch_file) and os.path.exists(segment_label_file):
                            # print(f"      Segment {segment_filename_base} 已存在，跳过。") # 输出可能过多
                            num_segments_in_file += 1
                            all_segment_metadata.append({
                                'segment_id': segment_filename_base,
                                'label': info['label'],
                                'subject_id': subject_id,
                                'source_file': fname
                            })
                            continue

                        # 保存 segment 数据和标签 (标签只有一个值)
                        np.save(segment_epoch_file, segment)
                        np.save(segment_label_file, np.array([info['label']], dtype=np.int64))  # 保存为包含单个标签的数组
                        num_segments_in_file += 1

                        # 收集元数据
                        all_segment_metadata.append({
                            'segment_id': segment_filename_base,  # 使用 segment ID 作为主键
                            'label': info['label'],
                            'subject_id': subject_id,  # 记录原始 subject ID
                            'source_file': fname  # 记录来源文件
                        })

                    logger.info(f"    从文件 {fname} 生成了 {num_segments_in_file} 个片段。")
                    subject_segment_count += num_segments_in_file

                except Exception as e:
                    logger.error(f"  处理文件 {fpath} 时出错: {e}")
                    subject_processed_successfully = False
                    # 可以选择 break 掉这个病人的处理，或者继续处理其他文件
                    continue  # 继续处理该病人的下一个文件

            if subject_processed_successfully and subject_segment_count > 0:
                processed_subjects.append(subject_id)
            elif not subject_processed_successfully:
                failed_subjects.append(subject_id)
            else:  # 成功处理但没有生成 segment (可能所有文件都太短)
                logger.warning(f"被试 {subject_id} 的所有文件处理完毕，但未生成任何有效片段。")
                failed_subjects.append(subject_id)  # 算作失败，因为没有数据输出

        logger.info("\n--- 预处理总结 ---")
        logger.info(f"成功处理并生成片段的被试: {processed_subjects}")
        if failed_subjects:
            logger.warning(f"处理失败或未生成任何片段的被试: {failed_subjects}")

        # --- 保存元数据 ---
        if all_segment_metadata:
            metadata_df = pd.DataFrame(all_segment_metadata)
            metadata_path = os.path.join(self.processed_data_dir, "segment_metadata.csv")
            try:
                metadata_df.to_csv(metadata_path, index=False, encoding='utf-8-sig')
                logger.info(f"所有片段的元数据已保存至: {metadata_path}")
            except Exception as e:
                logger.error(f"保存元数据文件失败: {e}")
        else:
            logger.warning("没有生成任何片段，未保存元数据文件。")

        # --- 保存 dataset_info.json (可以简化，因为 Dataloader 会加载 segment 文件) ---
        # 这里的元数据主要是为了记录配置和确认处理了哪些病人
        self.save_metadata(processed_subjects)

        return processed_subjects, failed_subjects

    # --- 覆盖基类的 save_metadata，内容可以简化 ---
    def save_metadata(self, processed_subjects):
        """ Saves metadata about the preprocessing run. """
        metadata_path = os.path.join(self.processed_data_dir, "dataset_info.json")
        # 尝试获取分段长度和采样率信息
        final_sfreq = getattr(self.dataset_conf, 'sfreq', None)
        n_times = None
        n_channels = None
        epoching_conf = getattr(self.preprocess_conf, 'epoching', None)
        if epoching_conf and final_sfreq:
            duration = getattr(epoching_conf, 'epoch_duration', None)
            if duration: n_times = int(duration * final_sfreq)
        # 尝试从第一个 segment 文件获取通道数
        try:
            first_seg_file = next(f for f in os.listdir(self.processed_data_dir) if f.endswith("_epochs.npy"))
            d = np.load(os.path.join(self.processed_data_dir, first_seg_file))
            n_channels = d.shape[0]  # NPY 保存的是 (C, L)
        except (StopIteration, IndexError, Exception):
            pass

        info = {
            "dataset_name": self.dataset_conf.name,
            "preprocessing_method_name": self.preprocess_conf.name,
            "processed_subjects_original_ids": processed_subjects,  # 记录成功处理的原始ID
            "config": {
                "dataset_conf": vars(self.dataset_conf),
                "preprocess_conf": vars(self.preprocess_conf)
            },
            "output_format": {
                "description": "Each segment saved as separate _epochs.npy and _labels.npy file.",
                "filename_base": "{subject_id}_seg{NNNNNN}",
                "epochs_shape": f"({n_channels or 'Unknown'}, {n_times or 'Unknown'})",
                "labels_shape": "(1,)"  # 因为每个 label 文件只存一个标签
            },
            "original_sfreq": final_sfreq,
            "processed_sfreq": final_sfreq  # 假设没有重采样
        }
        with open(metadata_path, 'w') as f:
            json.dump(info, f, indent=4, default=str)
        logger.info(f"数据集信息 (dataset_info.json) 已保存。")
