# -*- coding: utf-8 -*-

import serial
import struct
import crc
import numpy as np
import mne  # <--- 使用MNE库
import time
import os

# --- 用户配置区 (保持不变) ---
EDF_FILE_PATH = 'Patient_002_Sess01_ictal_01.edf'  # <--- 你的文件名
SERIAL_PORT = 'COM10'
BAUDRATE = 115200
EXPECTED_CHANNELS = 33
CHUNK_WIDTH = 512
SEND_INTERVAL_S =5

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
    主函数：读取EDF文件，分块，并通过串口发送。
    """
    if not os.path.exists(EDF_FILE_PATH):
        print(f"错误：EDF文件 '{EDF_FILE_PATH}' 不存在。")
        return

    # =================================================================
    #         !!! 使用 MNE-Python 读取 EDF 文件 !!!
    # =================================================================
    print(f"--- 正在使用 MNE-Python 读取EDF文件: {EDF_FILE_PATH} ---")
    try:
        # preload=True 表示将所有数据一次性加载到内存中
        # MNE会自动处理大多数头部不规范的问题
        raw = mne.io.read_raw_edf(EDF_FILE_PATH, preload=True, verbose='WARNING')

        # 从MNE的raw对象中获取数据，单位是伏特(V)
        # MNE返回的数据形状是 (通道数, 采样点数)
        eeg_data_full_volts = raw.get_data()

        # MNE读取的数据单位是V，而原始EDF文件通常存储的是uV。
        # 如果你的下游设备需要uV，最好进行转换。如果不需要，可以跳过。
        # 1 V = 1,000,000 uV
        # eeg_data_full = (eeg_data_full_volts * 1e6).astype(np.float32)

        # 如果你的设备处理的是原始的浮点电压值，直接使用即可
        eeg_data_full = eeg_data_full_volts.astype(np.float32)

        # 获取信息
        num_channels = raw.info['nchan']
        total_samples = raw.n_times

        print(f"文件信息: {num_channels}个通道, 每个通道{total_samples}个采样点。")

        if num_channels != EXPECTED_CHANNELS:
            print(f"警告：文件通道数({num_channels})与期望的通道数({EXPECTED_CHANNELS})不符！")

    except Exception as e:
        print(f"使用MNE读取EDF文件时出错: {e}")
        return

    print("--- EDF文件数据加载完成 ---")
    print(f"原始数据矩阵形状: {eeg_data_full.shape}")

    # =================================================================
    #         !!! 后续的串口准备和发送逻辑完全不变 !!!
    # =================================================================
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        print(f"成功打开串口 {SERIAL_PORT} @ {BAUDRATE} bps")
    except serial.SerialException as e:
        print(f"打开串口失败: {e}")
        return

    print(f"--- 开始将数据分割成 {EXPECTED_CHANNELS}x{CHUNK_WIDTH} 的数据块并发送 ---")
    num_chunks = (total_samples + CHUNK_WIDTH - 1) // CHUNK_WIDTH

    try:
        for i in range(num_chunks):
            start_index = i * CHUNK_WIDTH
            end_index = start_index + CHUNK_WIDTH
            eeg_chunk_np = eeg_data_full[:, start_index:end_index]

            payload_bytes = eeg_chunk_np.tobytes()
            if not payload_bytes: continue

            packet_to_send = create_packet(payload_bytes)
            ser.write(packet_to_send)

            print(f"-> 已发送数据块 {i + 1}/{num_chunks} | "
                  f"尺寸: {eeg_chunk_np.shape} | "
                  f"包总长: {len(packet_to_send)}字节")

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