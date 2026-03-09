# 在 Python 的 main 函数中
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

# ... (打开串口等) ...

# ====================================================================
#         !!! 使用新的、更安全的浮点数测试数据生成方法 !!!
# ====================================================================
# 打印出十六进制结果

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

# 在 Python 的 main 函数中

# ... (打开串口等) ...

# ====================================================================
#         !!! 终极简化测试：发送由同一个浮点数组成的数组 !!!
# ====================================================================

# 1. 定义矩阵形状和要重复的浮点数
eeg_matrix_shape = (33, 512)
num_elements = eeg_matrix_shape[0] * eeg_matrix_shape[1]
# 我们选择一个小数，因为它比整数更能测试浮点数的表示
repeating_float_value = 1.2345

# 2. 创建一个 numpy 数组，并用同一个值填充所有元素
eeg_matrix_np = np.full(num_elements, repeating_float_value, dtype=np.float32)

# 3. 将 numpy 数组转换为字节流 (这部分不变)
eeg_payload = eeg_matrix_np.tobytes()

# ====================================================================

print(f"准备发送极简测试包 (所有元素为 {repeating_float_value})")
print(f"原始数据字节数: {len(eeg_payload)}")

try:
    packet_to_send = create_packet(eeg_payload)

    # 为了调试，让我们打印出这个固定数据包的 CRC 值
    # 每次运行，这个值都应该是相同的
    sent_crc_bytes = packet_to_send[-2:]
    sent_crc_val = struct.unpack('<H', sent_crc_bytes)[0]
    print(f"Python 计算出的 CRC: {sent_crc_val} (0x{sent_crc_val:04X})")

    print(f"发送的包 (Hex, 前16字节): {packet_to_send[:66].hex(' ').upper()}")

    ser.write(packet_to_send)
    print(" -> 极简测试包已发送!")

except Exception as e:
    print(f"发送过程中出错: {e}")
finally:
    ser.close()
    print("串口已关闭")