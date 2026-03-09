import serial

# Configure serial port parameters
port = 'COM6'  # Replace with your actual COM port
baudrate = 9600
bytesize = 8
parity = 'N'
stopbits = 1

try:
    # Open the serial port
    ser = serial.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=bytesize,
        parity=parity,
        stopbits=stopbits,
        timeout=1
    )
    print(f"Successfully opened serial port {port}")

    # Example: Send some data
    data = [0x01, 0x02, 0x03, 0x04]
    print(f"Sending data: {data}")
    ser.write(bytes(data))

    # Close the serial port
    ser.close()
    print("Data sent, serial port closed")

except serial.SerialException as e:
    print(f"Serial port error: {e}")