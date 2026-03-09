import serial
import binascii


def main():
    # 配置串口参数
    port = 'COM5'
    baudrate = 9600
    timeout = 1

    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        print(f"已连接到串口 {port}，波特率 {baudrate}")
        print("正在接收数据（按Ctrl+C退出）...")
        print("所有数据以十六进制格式显示：")
        print("-" * 50)

        while True:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)

                # 将所有数据转换为十六进制，每字节显示为两位
                hex_data = binascii.hexlify(data).decode('ascii')

                # 格式化显示：每字节用空格分隔，每16字节换行
                formatted_output = ""
                for i in range(0, len(hex_data), 2):
                    if i > 0 and i % 32 == 0:  # 每16字节换行
                        formatted_output += '\n'
                    formatted_output += hex_data[i:i + 2] + " "

                print(formatted_output, end='', flush=True)

    except serial.SerialException as e:
        print(f"串口错误: {e}")
    except KeyboardInterrupt:
        print("\n程序退出")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()