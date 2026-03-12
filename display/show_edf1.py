import os
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, Slider
import mne


class EDFBrowser:
    def __init__(self, eeg_root, file_page_size=12, ch_page_size=14, preload=False):
        self.eeg_root = Path(eeg_root)
        self.file_page_size = file_page_size
        self.ch_page_size = ch_page_size
        self.preload = preload

        self.edf_files = sorted(self.eeg_root.rglob("*.edf"))
        if not self.edf_files:
            raise FileNotFoundError(f"在 {self.eeg_root} 下没有找到任何 .edf 文件")

        self.file_page = 0
        self.ch_page = 0
        self.selected_file_idx = 0
        self.selected_ch_idx = 0

        self.raw = None
        self.current_channels = []
        self.duration = 10.0   # 默认显示 10 秒
        self.t_start = 0.0

        self.fig = None
        self.ax_plot = None
        self.ax_file_radio = None
        self.ax_ch_radio = None
        self.file_radio = None
        self.ch_radio = None
        self.slider = None

        self.btn_file_prev = None
        self.btn_file_next = None
        self.btn_ch_prev = None
        self.btn_ch_next = None
        self.btn_zoom_in = None
        self.btn_zoom_out = None

        self._build_ui()
        self._load_file(self.selected_file_idx)
        self._update_file_radio()
        self._update_channel_radio()
        self._plot_current()

    def _build_ui(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title("EDF EEG Browser")

        # 主绘图区
        self.ax_plot = self.fig.add_axes([0.30, 0.18, 0.68, 0.75])

        # 文件选择区
        self.ax_file_radio = self.fig.add_axes([0.02, 0.30, 0.22, 0.55])
        self.ax_file_radio.set_title("EDF Files", fontsize=11)

        # 导联选择区
        self.ax_ch_radio = self.fig.add_axes([0.02, 0.02, 0.22, 0.22])
        self.ax_ch_radio.set_title("Channels", fontsize=11)

        # 文件翻页按钮
        ax_file_prev = self.fig.add_axes([0.02, 0.87, 0.10, 0.05])
        ax_file_next = self.fig.add_axes([0.14, 0.87, 0.10, 0.05])
        self.btn_file_prev = Button(ax_file_prev, "File Prev")
        self.btn_file_next = Button(ax_file_next, "File Next")
        self.btn_file_prev.on_clicked(self._on_file_prev)
        self.btn_file_next.on_clicked(self._on_file_next)

        # 导联翻页按钮
        ax_ch_prev = self.fig.add_axes([0.02, 0.25, 0.10, 0.04])
        ax_ch_next = self.fig.add_axes([0.14, 0.25, 0.10, 0.04])
        self.btn_ch_prev = Button(ax_ch_prev, "Ch Prev")
        self.btn_ch_next = Button(ax_ch_next, "Ch Next")
        self.btn_ch_prev.on_clicked(self._on_ch_prev)
        self.btn_ch_next.on_clicked(self._on_ch_next)

        # 时间滑条
        ax_slider = self.fig.add_axes([0.35, 0.08, 0.55, 0.03])
        self.slider = Slider(ax_slider, "Start(s)", 0.0, 1.0, valinit=0.0, valstep=0.01)
        self.slider.on_changed(self._on_slider_change)

        # 缩放按钮
        ax_zoom_in = self.fig.add_axes([0.92, 0.08, 0.03, 0.04])
        ax_zoom_out = self.fig.add_axes([0.96, 0.08, 0.03, 0.04])
        self.btn_zoom_in = Button(ax_zoom_in, "+")
        self.btn_zoom_out = Button(ax_zoom_out, "-")
        self.btn_zoom_in.on_clicked(self._on_zoom_in)
        self.btn_zoom_out.on_clicked(self._on_zoom_out)

    def _short_file_label(self, file_path):
        rel = file_path.relative_to(self.eeg_root)
        s = str(rel)
        return s if len(s) <= 40 else "..." + s[-37:]

    def _load_file(self, file_idx):
        file_idx = max(0, min(file_idx, len(self.edf_files) - 1))
        self.selected_file_idx = file_idx
        edf_path = self.edf_files[file_idx]

        if self.raw is not None:
            del self.raw

        self.raw = mne.io.read_raw_edf(
            str(edf_path),
            preload=self.preload,
            verbose="ERROR"
        )

        self.current_channels = self.raw.ch_names
        self.selected_ch_idx = min(self.selected_ch_idx, len(self.current_channels) - 1)
        self.ch_page = self.selected_ch_idx // self.ch_page_size

        total_time = self.raw.times[-1] if len(self.raw.times) > 0 else 1.0
        max_start = max(0.0, total_time - self.duration)

        if self.t_start > max_start:
            self.t_start = max_start

        self.slider.valmin = 0.0
        self.slider.valmax = max(1e-6, max_start if max_start > 0 else 1.0)
        self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)
        self.slider.set_val(self.t_start)

    def _update_file_radio(self):
        self.ax_file_radio.clear()
        self.ax_file_radio.set_title(
            f"EDF Files ({self.file_page + 1}/{math.ceil(len(self.edf_files) / self.file_page_size)})",
            fontsize=11
        )

        start = self.file_page * self.file_page_size
        end = min(start + self.file_page_size, len(self.edf_files))
        labels = [self._short_file_label(p) for p in self.edf_files[start:end]]

        active = self.selected_file_idx - start
        active = active if 0 <= active < len(labels) else 0

        self.file_radio = RadioButtons(self.ax_file_radio, labels, active=active)
        for txt in self.file_radio.labels:
            txt.set_fontsize(9)

        self.file_radio.on_clicked(self._on_file_selected)
        self.fig.canvas.draw_idle()

    def _update_channel_radio(self):
        self.ax_ch_radio.clear()
        total_pages = max(1, math.ceil(len(self.current_channels) / self.ch_page_size))
        self.ax_ch_radio.set_title(
            f"Channels ({self.ch_page + 1}/{total_pages})",
            fontsize=11
        )

        start = self.ch_page * self.ch_page_size
        end = min(start + self.ch_page_size, len(self.current_channels))
        labels = self.current_channels[start:end]

        active = self.selected_ch_idx - start
        active = active if 0 <= active < len(labels) else 0

        self.ch_radio = RadioButtons(self.ax_ch_radio, labels, active=active)
        for txt in self.ch_radio.labels:
            txt.set_fontsize(9)

        self.ch_radio.on_clicked(self._on_channel_selected)
        self.fig.canvas.draw_idle()

    def _plot_current(self):
        self.ax_plot.clear()

        ch_name = self.current_channels[self.selected_ch_idx]
        sfreq = self.raw.info["sfreq"]
        total_time = self.raw.times[-1] if len(self.raw.times) > 0 else 0

        start_samp = int(self.t_start * sfreq)
        stop_samp = int(min((self.t_start + self.duration) * sfreq, len(self.raw.times)))

        data, times = self.raw[self.selected_ch_idx, start_samp:stop_samp]
        data = data[0]
        times = times

        # 转成更直观的微伏
        data_uV = data * 1e6

        self.ax_plot.plot(times, data_uV, linewidth=0.8)
        self.ax_plot.set_xlabel("Time (s)")
        self.ax_plot.set_ylabel("Amplitude (uV)")
        self.ax_plot.grid(True, alpha=0.3)

        file_name = self.edf_files[self.selected_file_idx].name
        self.ax_plot.set_title(
            f"File: {file_name}\n"
            f"Channel: {ch_name} | "
            f"Window: {self.t_start:.2f}s - {min(self.t_start + self.duration, total_time):.2f}s / {total_time:.2f}s"
        )

        self.fig.canvas.draw_idle()

    def _on_file_selected(self, label):
        start = self.file_page * self.file_page_size
        end = min(start + self.file_page_size, len(self.edf_files))
        labels = [self._short_file_label(p) for p in self.edf_files[start:end]]

        local_idx = labels.index(label)
        self.selected_file_idx = start + local_idx

        self.selected_ch_idx = 0
        self.ch_page = 0
        self.t_start = 0.0

        self._load_file(self.selected_file_idx)
        self._update_channel_radio()
        self._plot_current()

    def _on_channel_selected(self, label):
        start = self.ch_page * self.ch_page_size
        labels = self.current_channels[start:start + self.ch_page_size]
        local_idx = labels.index(label)
        self.selected_ch_idx = start + local_idx
        self._plot_current()

    def _on_slider_change(self, value):
        self.t_start = float(value)
        self._plot_current()

    def _on_file_prev(self, event):
        if self.file_page > 0:
            self.file_page -= 1
            self._update_file_radio()

    def _on_file_next(self, event):
        max_page = math.ceil(len(self.edf_files) / self.file_page_size) - 1
        if self.file_page < max_page:
            self.file_page += 1
            self._update_file_radio()

    def _on_ch_prev(self, event):
        if self.ch_page > 0:
            self.ch_page -= 1
            self._update_channel_radio()

    def _on_ch_next(self, event):
        max_page = math.ceil(len(self.current_channels) / self.ch_page_size) - 1
        if self.ch_page < max_page:
            self.ch_page += 1
            self._update_channel_radio()

    def _on_zoom_in(self, event):
        self.duration = max(1.0, self.duration / 2.0)
        self._load_file(self.selected_file_idx)
        self._plot_current()

    def _on_zoom_out(self, event):
        total_time = self.raw.times[-1] if len(self.raw.times) > 0 else self.duration
        self.duration = min(total_time, self.duration * 2.0)
        self._load_file(self.selected_file_idx)
        self._plot_current()

    def show(self):
        plt.show()


if __name__ == "__main__":
    # 把这里改成你的 eeg 文件夹路径
    eeg_root = "data/epilepsy_eeg"

    browser = EDFBrowser(
        eeg_root=eeg_root,
        file_page_size=12,   # 每页显示多少个 EDF
        ch_page_size=12,     # 每页显示多少个导联
        preload=False        # 文件很大时建议 False
    )
    browser.show()
