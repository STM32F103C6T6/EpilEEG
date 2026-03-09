import serial

# 配置串口参数
port = '/dev/ttyUSB0'  # Linux 上的串口路径
baudrate = 9600        # 波特率
bytesize = 8           # 数据位
parity = 'N'           # 无校验
stopbits = 1           # 停止位

try:
    # 打开串口
    ser = serial.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=bytesize,
        parity=parity,
        stopbits=stopbits,
        timeout=1
    )
    print(f"成功打开串口 {port}")

    # 接收数据
    while True:
        data = ser.read(4)
        if data:
            print(f"接收到数据: {[hex(byte) for byte in data]}")
    
    if data:
        print(f"接收到数据: {[hex(byte) for byte in data]}")
    else:
        print("未接收到数据")

    # 关闭串口
    ser.close()
    print("串口已关闭")

except serial.SerialException as e:
    print(f"串口错误: {e}")