from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("tkagg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import mne


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
        if len(self.items) == 0:
            return
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

        top = 0.95
        step = 0.90 / max(self.page_size, 1)

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

        if len(self.items) > 0:
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
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.canvas.manager.set_window_title("EDF EEG Browser with Metrics")

        self.ax_file_list = self.fig.add_axes([0.02, 0.56, 0.20, 0.34])
        self.ax_file_slider = self.fig.add_axes([0.225, 0.56, 0.012, 0.34])

        self.ax_ch_list = self.fig.add_axes([0.02, 0.18, 0.20, 0.26])
        self.ax_ch_slider = self.fig.add_axes([0.225, 0.18, 0.012, 0.26])

        self.ax_plot = self.fig.add_axes([0.28, 0.53, 0.50, 0.36])
        self.ax_psd = self.fig.add_axes([0.28, 0.17, 0.50, 0.24])
        self.ax_metrics = self.fig.add_axes([0.80, 0.17, 0.18, 0.72])

        self.ax_metrics.set_title("Metrics / Hints", fontsize=11)
        self.ax_metrics.axis("off")

        self.ax_time_slider = self.fig.add_axes([0.34, 0.08, 0.38, 0.03])

        ax_zoom_in = self.fig.add_axes([0.75, 0.06, 0.04, 0.05])
        ax_zoom_out = self.fig.add_axes([0.80, 0.06, 0.04, 0.05])

        self.btn_zoom_in = Button(ax_zoom_in, "+")
        self.btn_zoom_out = Button(ax_zoom_out, "-")
        self.btn_zoom_in.on_clicked(self.on_zoom_in)
        self.btn_zoom_out.on_clicked(self.on_zoom_out)

        file_labels = [self._format_file_label(p) for p in self.edf_files]
        self.file_list = ScrollableList(
            self.fig, self.ax_file_list, file_labels, "EDF Files",
            self.on_file_selected, page_size=12
        )

        self.file_slider = Slider(
            self.ax_file_slider, "", 0,
            max(0, len(file_labels) - self.file_list.page_size),
            valinit=0, valstep=1, orientation="vertical"
        )
        self.file_slider.on_changed(self.on_file_slider_changed)

        self.ch_list = ScrollableList(
            self.fig, self.ax_ch_list, [], "Channels",
            self.on_channel_selected, page_size=10
        )

        self.ch_slider = Slider(
            self.ax_ch_slider, "", 0, 1,
            valinit=0, valstep=1, orientation="vertical"
        )
        self.ch_slider.on_changed(self.on_ch_slider_changed)

        self.time_slider = Slider(
            self.ax_time_slider, "Start (s)", 0, 1,
            valinit=0, valstep=0.1
        )
        self.time_slider.on_changed(self.on_time_slider_changed)

    def _format_file_label(self, path):
        rel = path.relative_to(self.eeg_root)
        text = str(rel)
        return text if len(text) <= 32 else "..." + text[-29:]

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
        self.win_sec = max(1.0, self.win_sec / 2.0)
        self._reset_time_slider()
        self.plot_signal()

    def on_zoom_out(self, event):
        total_duration = self.raw.times[-1] if len(self.raw.times) > 0 else self.win_sec
        self.win_sec = min(total_duration, self.win_sec * 2.0)
        self._reset_time_slider()
        self.plot_signal()

    def _get_current_window(self):
        sfreq = float(self.raw.info["sfreq"])
        start_sample = int(self.start_sec * sfreq)
        stop_sample = int(min((self.start_sec + self.win_sec) * sfreq, len(self.raw.times)))

        data, times = self.raw[self.channel_index, start_sample:stop_sample]
        signal_uv = data[0] * 1e6
        return signal_uv, times, sfreq

    def _compute_band_powers(self, x, sfreq):
        x = np.asarray(x, dtype=float)
        if len(x) < 4:
            return None

        x = x - np.mean(x)
        n = len(x)

        freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
        fft_vals = np.fft.rfft(x)
        psd = (np.abs(fft_vals) ** 2) / max(n, 1)

        valid = freqs <= 45
        freqs = freqs[valid]
        psd = psd[valid]

        total_power = np.trapz(psd, freqs) + 1e-12

        def bandpower(fmin, fmax):
            mask = (freqs >= fmin) & (freqs < fmax)
            if not np.any(mask):
                return 0.0
            return np.trapz(psd[mask], freqs[mask])

        bands = {
            "delta": bandpower(0.5, 4),
            "theta": bandpower(4, 8),
            "alpha": bandpower(8, 13),
            "beta":  bandpower(13, 30),
            "gamma": bandpower(30, 45),
        }
        rel_bands = {k: v / total_power for k, v in bands.items()}

        if np.any(freqs >= 0.5):
            dominant_freq = freqs[np.argmax(psd)]
        else:
            dominant_freq = 0.0

        p = psd / (np.sum(psd) + 1e-12)
        spectral_entropy = -np.sum(p * np.log2(p + 1e-12)) / np.log2(len(p) + 1e-12)

        return {
            "freqs": freqs,
            "psd": psd,
            "dominant_freq": float(dominant_freq),
            "rel_bands": rel_bands,
            "spectral_entropy": float(spectral_entropy),
        }

    def _find_spike_candidates(self, x, sfreq):
        """
        只是粗略找“尖锐瞬变候选点”，不能替代临床判读。
        """
        x = np.asarray(x, dtype=float)
        if len(x) < 5:
            return np.array([], dtype=int)

        dx = np.diff(x, prepend=x[0])
        mad = np.median(np.abs(dx - np.median(dx))) + 1e-12
        zdx = np.abs(dx - np.median(dx)) / (1.4826 * mad)

        amp_thr = np.median(np.abs(x - np.median(x))) * 4.0 + 1e-12
        candidates = []

        refractory = int(0.08 * sfreq)  # 80 ms
        last_idx = -refractory

        for i in range(1, len(x) - 1):
            local_peak = (x[i] > x[i - 1] and x[i] > x[i + 1]) or (x[i] < x[i - 1] and x[i] < x[i + 1])
            if not local_peak:
                continue

            if zdx[i] < 5.0:
                continue

            if abs(x[i] - np.median(x)) < amp_thr:
                continue

            if i - last_idx < refractory:
                continue

            candidates.append(i)
            last_idx = i

        return np.array(candidates, dtype=int)

    def _compute_metrics(self, signal_uv, sfreq):
        x = np.asarray(signal_uv, dtype=float)
        if len(x) < 4:
            return None

        mean_val = float(np.mean(x))
        std_val = float(np.std(x))
        rms_val = float(np.sqrt(np.mean(x ** 2)))
        ptp_val = float(np.ptp(x))
        line_length = float(np.sum(np.abs(np.diff(x))) / max(len(x) - 1, 1))

        bp = self._compute_band_powers(x, sfreq)
        spikes = self._find_spike_candidates(x, sfreq)

        slow_ratio = None
        if bp is not None:
            slow_ratio = bp["rel_bands"]["delta"] + bp["rel_bands"]["theta"]

        hints = []
        if bp is not None:
            if slow_ratio is not None and slow_ratio > 0.60:
                hints.append("Possible slowing: delta+theta ratio is high")
            if bp["dominant_freq"] < 8.0:
                hints.append("Dominant frequency is relatively slow")

        if ptp_val > 150:
            hints.append("High peak-to-peak amplitude in this window")

        if len(spikes) >= 3:
            hints.append("Several sharp transient candidates detected")

        return {
            "mean_uv": mean_val,
            "std_uv": std_val,
            "rms_uv": rms_val,
            "ptp_uv": ptp_val,
            "line_length": line_length,
            "band_info": bp,
            "spike_idx": spikes,
            "slow_ratio": slow_ratio,
            "hints": hints
        }

    def _draw_metrics_panel(self, metrics):
        self.ax_metrics.clear()
        self.ax_metrics.set_title("Metrics / Hints", fontsize=11)
        self.ax_metrics.axis("off")

        if metrics is None:
            self.ax_metrics.text(0.02, 0.98, "No metrics", va="top", fontsize=10)
            self.fig.canvas.draw_idle()
            return

        bp = metrics["band_info"]
        lines = [
            f"Peak-to-peak: {metrics['ptp_uv']:.2f} uV",
            f"STD:          {metrics['std_uv']:.2f} uV",
            f"RMS:          {metrics['rms_uv']:.2f} uV",
            f"Line length:  {metrics['line_length']:.2f}",
            f"Spike cand.:  {len(metrics['spike_idx'])}",
        ]

        if bp is not None:
            lines += [
                "",
                f"Dominant freq: {bp['dominant_freq']:.2f} Hz",
                f"Spec entropy:  {bp['spectral_entropy']:.3f}",
                "",
                "Relative band power:",
                f"Delta 0.5-4:   {bp['rel_bands']['delta']:.3f}",
                f"Theta 4-8:     {bp['rel_bands']['theta']:.3f}",
                f"Alpha 8-13:    {bp['rel_bands']['alpha']:.3f}",
                f"Beta 13-30:    {bp['rel_bands']['beta']:.3f}",
                f"Gamma 30-45:   {bp['rel_bands']['gamma']:.3f}",
                f"Slow ratio:    {metrics['slow_ratio']:.3f}",
            ]

        lines += ["", "Hints:"]
        if metrics["hints"]:
            lines += [f"- {x}" for x in metrics["hints"]]
        else:
            lines += ["- No obvious heuristic alert in this window"]

        note = [
            "",
            "Note:",
            "These are heuristic features only.",
            "They do not diagnose epilepsy."
        ]

        text = "\n".join(lines + note)
        self.ax_metrics.text(
            0.02, 0.98, text,
            va="top", ha="left",
            fontsize=9, family="monospace"
        )

    def plot_signal(self):
        self.ax_plot.clear()
        self.ax_psd.clear()

        signal_uv, times, sfreq = self._get_current_window()
        metrics = self._compute_metrics(signal_uv, sfreq)

        self.ax_plot.plot(times, signal_uv, linewidth=0.8)

        if metrics is not None and len(metrics["spike_idx"]) > 0:
            spike_times = times[metrics["spike_idx"]]
            spike_vals = signal_uv[metrics["spike_idx"]]
            self.ax_plot.scatter(spike_times, spike_vals, s=18, marker="o", label="Spike candidate")

        self.ax_plot.set_xlabel("Time (s)")
        self.ax_plot.set_ylabel("Amplitude (uV)")
        self.ax_plot.grid(True, alpha=0.3)

        total_duration = self.raw.times[-1] if len(self.raw.times) > 0 else 0
        self.ax_plot.set_title(
            f"File: {self.edf_files[self.file_index].name}\n"
            f"Channel: {self.channels[self.channel_index]} | "
            f"Window: {self.start_sec:.2f}s - {min(self.start_sec + self.win_sec, total_duration):.2f}s / {total_duration:.2f}s"
        )

        if metrics is not None and len(metrics["spike_idx"]) > 0:
            self.ax_plot.legend(loc="upper right", fontsize=8)

        if metrics is not None and metrics["band_info"] is not None:
            freqs = metrics["band_info"]["freqs"]
            psd = metrics["band_info"]["psd"]
            self.ax_psd.plot(freqs, psd, linewidth=1.0)
            self.ax_psd.set_xlim(0, 45)
            self.ax_psd.set_xlabel("Frequency (Hz)")
            self.ax_psd.set_ylabel("Power")
            self.ax_psd.set_title("Power Spectrum")
            self.ax_psd.grid(True, alpha=0.3)

            for f in [4, 8, 13, 30]:
                self.ax_psd.axvline(f, linestyle="--", linewidth=0.8, alpha=0.6)

        self._draw_metrics_panel(metrics)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == "__main__":
    eeg_folder = "../data/epilepsy_eeg"   # 改成你的 EEG 根目录
    viewer = EDFViewer(eeg_folder)
    viewer.show()
