# -*- coding: utf-8 -*-

import serial
import struct
import crc
import numpy as np
import pyedflib  # 引入读取EDF文件的库
import time
import os

# --- 用户配置区 ---
# 请将你的EDF文件放在与此脚本相同的目录下，或提供完整路径
EDF_FILE_PATH = 'Patient_002_Sess01_interictal_01.edf'  # <--- 修改这里：你的EDF文件名
SERIAL_PORT = 'COM10'  # <--- 修改这里：你的串口号
BAUDRATE = 115200  # <--- 修改这里：波特率
EXPECTED_CHANNELS = 33  # 期望的EDF通道数
CHUNK_WIDTH = 512  # 每个数据块的宽度（采样点数）
SEND_INTERVAL_S = 0.1  # (可选) 发送每个包之间的延时，单位秒

# --- 协议常量 ---
SOF = b'\xAA\x55'


def create_packet(payload: bytes) -> bytes:
    """
    根据协议，将数据负载打包成一个完整的数据包。
    - 帧头 (SOF): 2字节
    - 长度 (Length): 4字节, 小端模式
    - 数据 (Payload): N字节
    - 校验 (CRC): 2字节, CRC-16-MODBUS
    """
    length = len(payload)
    # 使用 '<I' 将长度打包成4字节小端无符号整数
    length_bytes = struct.pack('<I', length)

    # 需要进行CRC校验的数据是“长度 + 负载”
    data_to_checksum = length_bytes + payload

    # 计算CRC-16-MODBUS校验和
    calculator = crc.Calculator(crc.Crc16.MODBUS)
    checksum = calculator.checksum(data_to_checksum)
    checksum_bytes = struct.pack('<H', checksum)  # CRC是2字节

    # 拼接完整的数据包
    packet = SOF + length_bytes + payload + checksum_bytes
    return packet


def main():
    """
    主函数：读取EDF文件，分块，并通过串口发送。
    """
    # 1. 检查EDF文件是否存在
    if not os.path.exists(EDF_FILE_PATH):
        print(f"错误：EDF文件 '{EDF_FILE_PATH}' 不存在。请检查文件名和路径。")
        return

    # 2. 读取EDF文件并加载数据
    print(f"--- 正在读取EDF文件: {EDF_FILE_PATH} ---")
    try:
        reader = pyedflib.EdfReader(EDF_FILE_PATH)

        num_channels = reader.signals_in_file
        total_samples = reader.getNSamples()[0]  # 获取第一个通道的总采样点数

        print(f"文件信息: {num_channels}个通道, 每个通道{total_samples}个采样点。")

        if num_channels != EXPECTED_CHANNELS:
            print(f"警告：文件通道数({num_channels})与期望的通道数({EXPECTED_CHANNELS})不符！")
            # 根据需求，你可以在这里选择退出或继续
            # return

        # 创建一个列表来存储所有通道的数据
        all_signals = []
        for i in range(num_channels):
            # 将每个通道的数据读取为numpy数组并添加到列表中
            all_signals.append(reader.readSignal(i))

        # 将列表转换为一个大的2D numpy数组，形状为 (33, total_samples)
        eeg_data_full = np.array(all_signals, dtype=np.float32)

    except Exception as e:
        print(f"读取EDF文件时出错: {e}")
        return
    finally:
        # 确保关闭文件句柄
        if 'reader' in locals() and reader:
            reader.close()

    print("--- EDF文件数据加载完成 ---")
    print(f"原始数据矩阵形状: {eeg_data_full.shape}")

    # 3. 准备串口
    ser = None  # 初始化为None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        print(f"成功打开串口 {SERIAL_PORT} @ {BAUDRATE} bps")
    except serial.SerialException as e:
        print(f"打开串口失败: {e}")
        return

    # 4. 数据分块并循环发送
    print(f"--- 开始将数据分割成 {EXPECTED_CHANNELS}x{CHUNK_WIDTH} 的数据块并发送 ---")

    num_chunks = (total_samples + CHUNK_WIDTH - 1) // CHUNK_WIDTH  # 向上取整计算总块数

    try:
        for i in range(num_chunks):
            start_index = i * CHUNK_WIDTH
            end_index = start_index + CHUNK_WIDTH

            # 使用numpy切片获取数据块，形状为 (33, 512) 或最后一块可能更小
            # numpy的切片会自动处理边界，如果end_index超出范围，它会取到数组末尾
            eeg_chunk_np = eeg_data_full[:, start_index:end_index]

            # 将numpy数组块转换为字节流 (payload)
            payload_bytes = eeg_chunk_np.tobytes()

            # 如果数据块为空（虽然在正常逻辑下不应发生），则跳过
            if not payload_bytes:
                continue

            # 使用协议函数创建完整的数据包
            packet_to_send = create_packet(payload_bytes)

            # 发送数据
            ser.write(packet_to_send)

            print(f"-> 已发送数据块 {i + 1}/{num_chunks} | "
                  f"尺寸: {eeg_chunk_np.shape} | "
                  f"包总长: {len(packet_to_send)}字节")

            # 等待一小段时间，给接收方处理时间，避免发送过快导致对方丢包
            if SEND_INTERVAL_S > 0:
                time.sleep(SEND_INTERVAL_S)

        print("\n--- 所有数据块发送完毕 ---")

    except Exception as e:
        print(f"发送过程中出错: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("串口已关闭")


if __name__ == '__main__':
    main()