from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import mne
import matplotlib
matplotlib.use('tkagg')
class ScrollableList:
    def __init__(self, fig, ax, items, title, on_select, page_size=12):
        self.fig = fig
        self.ax = ax
        self.items = items
        self.title = title
        self.on_select = on_select
        self.page_size = page_size
        self.start_index = 0
        self.selected_index = 0
        self.text_artists = []

        self.ax.set_title(self.title, fontsize=11)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)

        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.cid_scroll = self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

        self.draw()

    def set_items(self, items, selected_index=0):
        self.items = items
        self.selected_index = min(max(0, selected_index), max(0, len(items) - 1))
        self.start_index = min(self.start_index, max(0, len(items) - self.page_size))
        if self.selected_index < self.start_index:
            self.start_index = self.selected_index
        elif self.selected_index >= self.start_index + self.page_size:
            self.start_index = max(0, self.selected_index - self.page_size + 1)
        self.draw()

    def set_start(self, start_index):
        max_start = max(0, len(self.items) - self.page_size)
        self.start_index = int(np.clip(start_index, 0, max_start))
        self.draw()

    def set_selected(self, index):
        self.selected_index = int(np.clip(index, 0, len(self.items) - 1))
        if self.selected_index < self.start_index:
            self.start_index = self.selected_index
        elif self.selected_index >= self.start_index + self.page_size:
            self.start_index = self.selected_index - self.page_size + 1
        self.draw()

    def draw(self):
        self.ax.clear()
        self.ax.set_title(self.title, fontsize=11)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)

        self.text_artists = []

        visible_items = self.items[self.start_index:self.start_index + self.page_size]
        n = max(len(visible_items), 1)

        top = 0.95
        step = 0.90 / self.page_size

        for i, item in enumerate(visible_items):
            global_idx = self.start_index + i
            y = top - i * step

            is_selected = (global_idx == self.selected_index)
            bbox = dict(
                facecolor="#dbeafe" if is_selected else "#f3f4f6",
                edgecolor="#93c5fd" if is_selected else "#d1d5db",
                boxstyle="round,pad=0.2"
            )

            txt = self.ax.text(
                0.02, y, item,
                transform=self.ax.transAxes,
                fontsize=9,
                va="top",
                ha="left",
                bbox=bbox,
                clip_on=True
            )
            self.text_artists.append((txt, global_idx))

        self.ax.text(
            0.98, 0.02,
            f"{min(len(self.items), self.start_index + 1)}-{min(len(self.items), self.start_index + self.page_size)} / {len(self.items)}",
            transform=self.ax.transAxes,
            fontsize=8,
            ha="right",
            va="bottom"
        )

        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return

        for txt, idx in self.text_artists:
            contains, _ = txt.contains(event)
            if contains:
                self.selected_index = idx
                self.draw()
                self.on_select(idx)
                return

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == "up":
            self.set_start(self.start_index - 1)
        elif event.button == "down":
            self.set_start(self.start_index + 1)


class EDFViewer:
    def __init__(self, eeg_root):
        self.eeg_root = Path(eeg_root)
        self.edf_files = sorted(self.eeg_root.rglob("*.edf"))

        if not self.edf_files:
            raise ValueError(f"No EDF files found under: {self.eeg_root}")

        self.file_index = 0
        self.channel_index = 0
        self.raw = None
        self.channels = []

        self.win_sec = 10.0
        self.start_sec = 0.0

        self._build_ui()
        self.load_file(0)

    def _build_ui(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title("EDF EEG Browser")

        self.ax_file_list = self.fig.add_axes([0.03, 0.52, 0.22, 0.38])
        self.ax_file_slider = self.fig.add_axes([0.26, 0.52, 0.015, 0.38])

        self.ax_ch_list = self.fig.add_axes([0.03, 0.12, 0.22, 0.28])
        self.ax_ch_slider = self.fig.add_axes([0.26, 0.12, 0.015, 0.28])

        self.ax_plot = self.fig.add_axes([0.32, 0.20, 0.65, 0.68])
        self.ax_time_slider = self.fig.add_axes([0.36, 0.08, 0.50, 0.03])

        ax_zoom_in = self.fig.add_axes([0.88, 0.06, 0.04, 0.05])
        ax_zoom_out = self.fig.add_axes([0.93, 0.06, 0.04, 0.05])

        self.btn_zoom_in = Button(ax_zoom_in, "+")
        self.btn_zoom_out = Button(ax_zoom_out, "-")
        self.btn_zoom_in.on_clicked(self.on_zoom_in)
        self.btn_zoom_out.on_clicked(self.on_zoom_out)

        file_labels = [self._format_file_label(p) for p in self.edf_files]
        self.file_list = ScrollableList(
            self.fig,
            self.ax_file_list,
            file_labels,
            "EDF Files",
            self.on_file_selected,
            page_size=12
        )

        self.file_slider = Slider(
            self.ax_file_slider,
            "",
            0,
            max(0, len(file_labels) - self.file_list.page_size),
            valinit=0,
            valstep=1,
            orientation="vertical"
        )
        self.file_slider.on_changed(self.on_file_slider_changed)

        self.ch_list = ScrollableList(
            self.fig,
            self.ax_ch_list,
            [],
            "Channels",
            self.on_channel_selected,
            page_size=10
        )

        self.ch_slider = Slider(
            self.ax_ch_slider,
            "",
            0,
            1,
            valinit=0,
            valstep=1,
            orientation="vertical"
        )
        self.ch_slider.on_changed(self.on_ch_slider_changed)

        self.time_slider = Slider(
            self.ax_time_slider,
            "Start (s)",
            0,
            1,
            valinit=0,
            valstep=0.1
        )
        self.time_slider.on_changed(self.on_time_slider_changed)

    def _format_file_label(self, path):
        rel = path.relative_to(self.eeg_root)
        text = str(rel)
        return text if len(text) <= 36 else "..." + text[-33:]

    def load_file(self, index):
        self.file_index = index
        edf_path = self.edf_files[index]

        print(f"Loading: {edf_path}")
        self.raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        self.channels = self.raw.ch_names
        self.channel_index = 0
        self.start_sec = 0.0

        self.file_list.set_selected(index)

        self.ch_list.set_items(self.channels, selected_index=0)
        self._reset_ch_slider()

        self._reset_time_slider()
        self.plot_signal()

    def _reset_ch_slider(self):
        max_start = max(0, len(self.channels) - self.ch_list.page_size)
        self.ch_slider.valmin = 0
        self.ch_slider.valmax = max(1, max_start)
        self.ch_slider.ax.set_ylim(self.ch_slider.valmin, self.ch_slider.valmax)
        self.ch_slider.set_val(self.ch_list.start_index)

    def _reset_time_slider(self):
        total_duration = self.raw.times[-1] if len(self.raw.times) > 0 else 1
        max_start = max(0, total_duration - self.win_sec)

        self.time_slider.valmin = 0
        self.time_slider.valmax = max(1, max_start)
        self.time_slider.ax.set_xlim(self.time_slider.valmin, self.time_slider.valmax)
        self.time_slider.set_val(self.start_sec)

    def on_file_selected(self, index):
        self.load_file(index)

    def on_channel_selected(self, index):
        self.channel_index = index
        self.plot_signal()

    def on_file_slider_changed(self, val):
        self.file_list.set_start(int(val))

    def on_ch_slider_changed(self, val):
        self.ch_list.set_start(int(val))

    def on_time_slider_changed(self, val):
        self.start_sec = float(val)
        self.plot_signal()

    def on_zoom_in(self, event):
        self.win_sec = max(1.0, self.win_sec / 2)
        self._reset_time_slider()
        self.plot_signal()

    def on_zoom_out(self, event):
        total_duration = self.raw.times[-1] if len(self.raw.times) > 0 else self.win_sec
        self.win_sec = min(total_duration, self.win_sec * 2)
        self._reset_time_slider()
        self.plot_signal()

    def plot_signal(self):
        self.ax_plot.clear()

        sfreq = self.raw.info["sfreq"]
        start_sample = int(self.start_sec * sfreq)
        stop_sample = int(min((self.start_sec + self.win_sec) * sfreq, len(self.raw.times)))

        data, times = self.raw[self.channel_index, start_sample:stop_sample]
        data_uv = data[0] * 1e6

        self.ax_plot.plot(times, data_uv, linewidth=0.8)
        self.ax_plot.set_xlabel("Time (s)")
        self.ax_plot.set_ylabel("Amplitude (uV)")
        self.ax_plot.grid(True, alpha=0.3)

        total_duration = self.raw.times[-1] if len(self.raw.times) > 0 else 0
        self.ax_plot.set_title(
            f"File: {self.edf_files[self.file_index].name}\n"
            f"Channel: {self.channels[self.channel_index]} | "
            f"Window: {self.start_sec:.2f}s - {min(self.start_sec + self.win_sec, total_duration):.2f}s / {total_duration:.2f}s"
        )

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == "__main__":
    eeg_folder = "../data/epilepsy_eeg"   # 改成你的 eeg 根目录
    viewer = EDFViewer(eeg_folder)
    viewer.show()
