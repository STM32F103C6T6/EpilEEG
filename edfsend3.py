# -*- coding: utf-8 -*-

import serial
import struct
import crc
import numpy as np
import mne
import time
import os
import onnxruntime as ort  # <--- 导入ONNX Runtime库

# --- 用户配置区 ---
EDF_FILE_PATH = 'Patient_002_Sess01_interictal_01.edf'
ONNX_MODEL_PATH = 'your_model.onnx'  # <--- 修改这里：你的ONNX模型文件名
SERIAL_PORT = 'COM10'
BAUDRATE = 115200
EXPECTED_CHANNELS = 33
CHUNK_WIDTH = 512
SEND_INTERVAL_S = 0.1

# --- 协议函数 create_packet() (保持不变) ---
SOF = b'\xAA\x55'


def create_packet(payload: bytes) -> bytes:
    length = len(payload)
    length_bytes = struct.pack('<I', length)
    data_to_checksum = length_bytes + payload
    calculator = crc.Calculator(crc.Crc16.MODBUS)
    checksum = calculator.checksum(data_to_checksum)
    checksum_bytes = struct.pack('<H', checksum)
    return SOF + length_bytes + payload + checksum_bytes


def main():
    """
    主函数：读取EDF，分块，进行ONNX推理，并通过串口发送。
    """
    # 1. 检查文件是否存在
    if not os.path.exists(EDF_FILE_PATH):
        print(f"错误：EDF文件 '{EDF_FILE_PATH}' 不存在。")
        return
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"错误：ONNX模型文件 '{ONNX_MODEL_PATH}' 不存在。")
        return

    # 2. 加载ONNX模型并创建推理会话
    print(f"--- 正在加载ONNX模型: {ONNX_MODEL_PATH} ---")
    try:
        ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
        # 获取模型的输入节点名称和期望的输入形状
        input_name = ort_session.get_inputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        print(f"模型加载成功。输入节点: '{input_name}', 期望形状: {input_shape}")
    except Exception as e:
        print(f"加载ONNX模型失败: {e}")
        return

    # 3. 使用 MNE-Python 读取 EDF 文件 (逻辑不变)
    print(f"--- 正在使用 MNE-Python 读取EDF文件: {EDF_FILE_PATH} ---")
    try:
        raw = mne.io.read_raw_edf(EDF_FILE_PATH, preload=True, verbose='WARNING')
        eeg_data_full = raw.get_data().astype(np.float32)
        num_channels, total_samples = eeg_data_full.shape
        print(f"文件信息: {num_channels}个通道, {total_samples}个采样点。")
        if num_channels != EXPECTED_CHANNELS:
            print(f"警告：文件通道数({num_channels})与期望的通道数({EXPECTED_CHANNELS})不符！")
    except Exception as e:
        print(f"读取EDF文件时出错: {e}")
        return

    # 4. 准备串口 (逻辑不变)
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        print(f"成功打开串口 {SERIAL_PORT} @ {BAUDRATE} bps")
    except serial.SerialException as e:
        print(f"打开串口失败: {e}")
        return

    # 5. 数据分块, 推理, 并循环发送
    print(f"\n--- 开始处理数据块 (推理 -> 发送) ---")
    num_chunks = (total_samples + CHUNK_WIDTH - 1) // CHUNK_WIDTH

    try:
        for i in range(num_chunks):
            print(f"\n----- 处理数据块 {i + 1}/{num_chunks} -----")

            # --- a. 获取数据块 ---
            start_index = i * CHUNK_WIDTH
            end_index = start_index + CHUNK_WIDTH
            eeg_chunk_np = eeg_data_full[:, start_index:end_index]

            # 检查最后一个块的宽度，如果宽度不等于512，可能需要特殊处理
            if eeg_chunk_np.shape[1] != CHUNK_WIDTH:
                print(f"警告: 最后一个数据块宽度为 {eeg_chunk_np.shape[1]}, 不等于 {CHUNK_WIDTH}。跳过此块的推理和发送。")
                # 根据你的模型要求，你可能需要填充(padding)这个块，或者直接跳过
                continue

            # --- b. 在Python端进行ONNX推理 (核心新增部分) ---

            # ONNX模型通常需要一个批次维度(batch dimension)，所以我们需要将
            # 输入数据从 (33, 512) 变形为 (1, 33, 512)
            # 注意: 如果你的模型输入是(1, 1, 33, 512)或(1, 33, 512, 1)等，请相应修改
            input_tensor = np.expand_dims(eeg_chunk_np, axis=0)  # 变为 (1, 33, 512)

            # 确保数据类型与模型匹配，通常是float32
            input_tensor = input_tensor.astype(np.float32)

            # 执行推理
            ort_inputs = {input_name: input_tensor}
            ort_outs = ort_session.run(None, ort_inputs)

            # 打印推理结果，ort_outs是一个列表，我们取第一个元素
            inference_result_py = ort_outs[0]
            print(f"Python端推理结果 (前5个值): {inference_result_py.flatten()[:5]}")
            # 你可以在这里打印更多值或者整个数组来进行详细对比
            # print(f"Python端完整推理结果: {inference_result_py}")

            # --- c. 发送数据 (逻辑不变) ---
            payload_bytes = eeg_chunk_np.tobytes()
            if not payload_bytes: continue

            packet_to_send = create_packet(payload_bytes)
            ser.write(packet_to_send)

            print(f"-> 数据块已发送 | 尺寸: {eeg_chunk_np.shape} | 包总长: {len(packet_to_send)}字节")

            if SEND_INTERVAL_S > 0:
                time.sleep(SEND_INTERVAL_S)

        print("\n--- 所有有效数据块处理完毕 ---")

    except Exception as e:
        print(f"处理过程中出错: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("串口已关闭")


if __name__ == '__main__':
    main()