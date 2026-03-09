import serial
import struct
import crc
import numpy as np

# --- 协议常量 ---
SOF = b'\xAA\x55'


def create_packet(payload: bytes) -> bytes:
    """
    根据【升级版】协议，将数据负载打包成一个完整的数据包。
    长度字段现在是4字节。
    """
    length = len(payload)

    # !!! 核心修改点: 使用 '<I' 来打包4字节的长度 !!!
    # '<' 表示小端, 'I' 表示无符号整型 (4字节)
    length_bytes = struct.pack('<I', length)

    # CRC计算部分逻辑不变，但现在包含了4字节的长度
    data_to_checksum = length_bytes + payload

    calculator = crc.Calculator(crc.Crc16.MODBUS)
    checksum = calculator.checksum(data_to_checksum)
    print(checksum)
    checksum_bytes = struct.pack('<H', checksum)  # CRC本身还是2字节

    # 拼接完整的数据包
    packet = SOF + length_bytes + payload + checksum_bytes

    return packet


def main():
    # ... (main 函数的其他部分无需修改) ...
    port = 'COM10'  # 请根据你的实际情况修改
    baudrate = 115200
    # 准备一个固定的测试数据
    length_bytes = struct.pack('<I', 4)
    payload_bytes = bytes([0x01, 0x02, 0x03, 0x04])
    data_to_checksum = length_bytes + payload_bytes

    calculator = crc.Calculator(crc.Crc16.MODBUS)
    checksum = calculator.checksum(data_to_checksum)

    # 打印出十六进制结果
    print(f"Python CRC for [04 00 00 00 01 02 03 04] is: 0x{checksum:04X}")
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"成功打开串口 {port} @ {baudrate} bps")
    except serial.SerialException as e:
        print(f"打开串口失败: {e}")
        return

    eeg_matrix_shape = (33, 512)
    eeg_matrix_np = np.arange(eeg_matrix_shape[0] * eeg_matrix_shape[1], dtype=np.float32).reshape(eeg_matrix_shape)

    eeg_payload = eeg_matrix_np.tobytes()

    # 检查一下长度，确认它现在可以被处理
    print(f"准备发送 EEG 矩阵，字节数: {len(eeg_payload)}")  # 输出: 67584

    try:
        packet_to_send = create_packet(eeg_payload)
        print(f"完整数据包大小: {len(packet_to_send)} bytes")  # SOF(2)+LEN(4)+PL(67584)+CRC(2) = 67592
        print(packet_to_send.hex(' ').upper())
        ser.write(packet_to_send)
        print(" -> EEG 数据包已发送!")

    except Exception as e:
        print(f"发送过程中出错: {e}")
    finally:
        ser.close()
        print("串口已关闭")


if __name__ == "__main__":
    main()