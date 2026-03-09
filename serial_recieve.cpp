#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>

int main() {
    int fd;                // 文件描述符
    struct termios options; // 串口配置结构体

    // 打开串口 /dev/ttyUSB0
    fd = open("/dev/ttyUSB0", O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) {
        perror("无法打开串口 /dev/ttyUSB0");
        return 1;
    }
    printf("成功打开串口 /dev/ttyUSB0\n");

    // 获取当前串口配置
    tcgetattr(fd, &options);

    // 设置波特率
    cfsetispeed(&options, B9600);
    cfsetospeed(&options, B9600);

    // 设置 8 数据位，无校验，1 停止位
    options.c_cflag &= ~CSIZE;   // 清除数据位设置
    options.c_cflag |= CS8;      // 8 数据位
    options.c_cflag &= ~PARENB;  // 无校验
    options.c_cflag &= ~CSTOPB;  // 1 停止位

    // 设置无流控制
    options.c_cflag &= ~CRTSCTS;  // 无硬件流控制
    options.c_iflag &= ~(IXON | IXOFF | IXANY); // 无软件流控制

    // 设置为原始模式
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;

    // 应用配置
    tcsetattr(fd, TCSANOW, &options);

    // 接收数据
    char buf[255];  // 接收缓冲区
    int n = read(fd, buf, sizeof(buf));
    if (n > 0) {
        printf("接收到 %d 个字节: ", n);
        for (int i = 0; i < n; i++) {
            printf("0x%02x ", (unsigned char)buf[i]);
        }
        printf("\n");
    } else {
        printf("未接收到数据\n");
    }

    // 关闭串口
    close(fd);
    printf("串口已关闭\n");

    return 0;
}