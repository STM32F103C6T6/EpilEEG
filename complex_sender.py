import serial
import struct
import crc
import time
import numpy as np  # 使用 numpy 来方便地创建大型数组

# --- 协议常量 ---
SOF = b'\xAA\x55'


def create_packet(payload: bytes) -> bytes:
    # (这个函数和之前一样，无需修改)
    length = len(payload)
    if length > 65535:
        raise ValueError("Payload size exceeds 65535 bytes")
    length_bytes = struct.pack('<H', length)
    data_to_checksum = length_bytes + payload
    calculator = crc.Calculator(crc.Crc16.MODBUS)
    checksum = calculator.checksum(data_to_checksum)
    checksum_bytes = struct.pack('<H', checksum)
    return SOF + length_bytes + payload + checksum_bytes


def main():
    # !!! 修改为你自己的串口号和波特率
    port = 'COM10'  # 发送端和接收端用不同的串口
    baudrate = 921600

    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"成功打开串口 {port} @ {baudrate} bps")
    except serial.SerialException as e:
        print(f"打开串口失败: {e}")
        return

    # !!! 核心修改点: 创建一个 33x512 的浮点数矩阵并打包 !!!
    # 1. 创建符合 ONNX 输入形状的 numpy 数组
    eeg_matrix_shape = (33, 512)
    # 用随机数或递增序列创建测试数据
    eeg_matrix_np = np.arange(eeg_matrix_shape[0] * eeg_matrix_shape[1], dtype=np.float32).reshape(eeg_matrix_shape)
    eeg_matrix_np[0, 0] = 1.23  # 修改几个值方便调试
    eeg_matrix_np[0, 1] = -4.56

    # 2. 将 numpy 数组转换为字节流
    # .tobytes() 会将其平铺成一维字节序列
    eeg_payload = eeg_matrix_np.tobytes()

    print(f"准备发送 EEG 矩阵，形状: {eeg_matrix_shape}, 字节数: {len(eeg_payload)}")

    try:
        packet_to_send = create_packet(eeg_payload)
        print(f"完整数据包大小: {len(packet_to_send)} bytes")
        ser.write(packet_to_send)
        print(" -> EEG 数据包已发送!")

    except Exception as e:
        print(f"发送过程中出错: {e}")
    finally:
        ser.close()
        print("串口已关闭")


if __name__ == "__main__":
    main()