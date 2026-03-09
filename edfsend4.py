# -*- coding: utf-8 -*-

import serial
import struct
import crc
import numpy as np
import time
import os
import onnxruntime as ort

# --- 用户配置区 ---
# 包含 (N, 33, 512) 形状数据的NPY文件
NPY_DATA_PATH = 'processed_data/epilepsy_eeg/epilepsy_filter_epoch/Patient_004_epochs.npy'
ONNX_MODEL_PATH = 'your_model.onnx'  # 你的ONNX模型文件名
SERIAL_PORT = 'COM10'
BAUDRATE = 115200*2
SEND_INTERVAL_S = 3  # 发送每个包之间的延时

# --- 协议函数 create_packet() (保持不变) ---
SOF = b'\xAA\x55'


def create_packet(payload: bytes) -> bytes:
    """根据协议，将数据负载打包成一个完整的数据包。"""
    length = len(payload)
    length_bytes = struct.pack('<I', length)
    data_to_checksum = length_bytes + payload
    calculator = crc.Calculator(crc.Crc16.MODBUS)
    checksum = calculator.checksum(data_to_checksum)
    checksum_bytes = struct.pack('<H', checksum)
    return SOF + length_bytes + payload + checksum_bytes


def main():
    """
    主函数：读取NPY文件中的数据块，对每个块进行推理，并通过串口发送。
    """
    # 1. 检查所有必需文件是否存在
    if not os.path.exists(NPY_DATA_PATH):
        print(f"错误：NPY数据文件 '{NPY_DATA_PATH}' 不存在。")
        # 为了方便测试，我们可以创建一个示例文件
        response = input("是否要创建一个示例NPY文件 (10个33x512的矩阵)? (y/n): ").lower()
        if response == 'y':
            print("正在创建示例文件...")
            sample_data = np.random.randn(10, 33, 512).astype(np.float32)
            np.save(NPY_DATA_PATH, sample_data)
            print(f"示例文件 '{NPY_DATA_PATH}' 创建成功。")
        else:
            return

    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"错误：ONNX模型文件 '{ONNX_MODEL_PATH}' 不存在。")
        return

    # 2. 加载ONNX模型
    print(f"--- 正在加载ONNX模型: {ONNX_MODEL_PATH} ---")
    try:
        ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
        input_name = ort_session.get_inputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        print(f"模型加载成功。输入节点: '{input_name}', 期望形状: {input_shape}")
    except Exception as e:
        print(f"加载ONNX模型失败: {e}")
        return

    # 3. 加载NPY数据文件
    print(f"--- 正在加载NPY数据文件: {NPY_DATA_PATH} ---")
    try:
        all_eeg_chunks = np.load(NPY_DATA_PATH)
        # 验证数据形状
        if all_eeg_chunks.ndim != 3 or all_eeg_chunks.shape[1] != 33 or all_eeg_chunks.shape[2] != 512:
            print(f"错误：NPY文件中的数据形状为 {all_eeg_chunks.shape}，不符合预期的 (N, 33, 512) 格式。")
            return
        num_chunks = all_eeg_chunks.shape[0]
        print(f"数据加载成功，共包含 {num_chunks} 个 33x512 的矩阵。")
    except Exception as e:
        print(f"加载NPY文件失败: {e}")
        return

    # 4. 准备串口
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        print(f"成功打开串口 {SERIAL_PORT} @ {BAUDRATE} bps")
    except serial.SerialException as e:
        print(f"打开串口失败: {e}")
        return

    # 5. 遍历所有数据块, 进行推理和发送
    print(f"\n--- 开始处理 {num_chunks} 个数据块 (推理 -> 发送) ---")

    try:
        # 直接遍历三维数组的第一维
        for i, eeg_chunk_np in enumerate(all_eeg_chunks):
            print(f"\n----- 处理数据块 {i + 1}/{num_chunks} -----")

            # 确保数据类型为float32
            eeg_chunk_np = eeg_chunk_np.astype(np.float32)

            # --- a. 在Python端进行ONNX推理 ---
            # 为模型添加批次维度: (33, 512) -> (1, 33, 512)
            input_tensor = np.expand_dims(eeg_chunk_np, axis=0)

            # 执行推理
            ort_inputs = {input_name: input_tensor}
            ort_outs = ort_session.run(None, ort_inputs)

            inference_result_py = ort_outs[0]
            print(f"Python端推理结果 (前5个值): {inference_result_py.flatten()[:5]}")
            if(inference_result_py.flatten()[1]>inference_result_py.flatten()[0]):
                print("special!")
            else:
                print("normal")

            # --- b. 发送数据 ---
            payload_bytes = eeg_chunk_np.tobytes()
            packet_to_send = create_packet(payload_bytes)

            ser.write(packet_to_send)
            print(f"-> 数据块已发送 | 尺寸: {eeg_chunk_np.shape} | 包总长: {len(packet_to_send)}字节")

            if SEND_INTERVAL_S > 0:
                time.sleep(SEND_INTERVAL_S)

        print(f"\n--- 所有 {num_chunks} 个数据块处理完毕 ---")

    except Exception as e:
        print(f"处理过程中出错: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("串口已关闭")


if __name__ == '__main__':
    main()