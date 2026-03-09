"""
EDF通道快速查看工具
"""
import os
import mne
from pathlib import Path
import sys


def quick_check_edf_channels(directory):
    """快速查看EDF文件通道"""
    print(f"🔍 快速扫描: {directory}")
    print("=" * 100)

    edf_files = list(Path(directory).rglob('*.edf'))

    if not edf_files:
        print("❌ 没有找到EDF文件")
        return

    print(f"找到 {len(edf_files)} 个EDF文件")
    print()

    # 按通道数分组统计
    channel_groups = {}

    for i, file_path in enumerate(edf_files):
        try:
            # 快速加载头信息
            raw = mne.io.read_raw_edf(str(file_path), preload=False, verbose=False)
            channels = raw.ch_names
            num_channels = len(channels)

            # 分组
            if num_channels not in channel_groups:
                channel_groups[num_channels] = {
                    'count': 0,
                    'example_channels': channels,
                    'files': []
                }

            channel_groups[num_channels]['count'] += 1
            channel_groups[num_channels]['files'].append(str(file_path))

            # 显示文件信息
            rel_path = os.path.relpath(file_path, directory)
            print(f"[{i + 1:3d}] {rel_path:<50} 通道: {num_channels:2d}")

            # 显示具体通道
            if num_channels <= 20:
                print(f"     {' | '.join(channels)}")
            else:
                print(f"     前?个: {' | '.join(channels[:])}...")
            print()

        except Exception as e:
            print(f"[{i + 1:3d}] ❌ {file_path.name:<50} 错误: {str(e)[:50]}...")
            print()

    # 打印分组统计
    print("=" * 100)
    print("📊 通道数分组统计:")
    print("=" * 100)

    for channels, info in sorted(channel_groups.items()):
        print(f"\n{channels} 通道文件 ({info['count']} 个):")
        print(f"示例通道: {' | '.join(info['example_channels'][:15])}")
        if len(info['example_channels']) > 15:
            print(f"         ... 还有 {len(info['example_channels']) - 15} 个通道")

        # 显示前3个文件
        print("示例文件:")
        for file in info['files'][:3]:
            print(f"  - {os.path.basename(file)}")
        if len(info['files']) > 3:
            print(f"  ... 还有 {len(info['files']) - 3} 个文件")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法: python edf_quick_check.py <目录路径>")
        print("示例: python edf_quick_check.py E:\\CHB-MIT\\")
        sys.exit(1)

    directory = sys.argv[1]
    quick_check_edf_channels(directory)