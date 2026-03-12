from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("tkagg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
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

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.draw()

    def set_items(self, items, selected_index=0):
        self.items = items
        if len(items) == 0:
            self.selected_index = 0
            self.start_index = 0
        else:
            self.selected_index = min(max(0, selected_index), len(items) - 1)
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


class MultiChannelEDFViewer:
    def __init__(self, eeg_root):
        self.eeg_root = Path(eeg_root)
        self.edf_files = sorted(self.eeg_root.rglob("*.edf"))
        if not self.edf_files:
            raise ValueError(f"No EDF files found under: {self.eeg_root}")

        self.file_index = 0
        self.raw_orig = None
        self.raw_proc = None
        self.channels = []
        self.auto_bads = []

        self.win_sec = 10.0
        self.start_sec = 0.0

        self.enable_bandpass = True
        self.enable_notch = True
        self.enable_avg_ref = True
        self.enable_interp = True

        self.l_freq = 0.5
        self.h_freq = 40.0
        self.notch_freq = 50.0

        self.channels_per_page = 8
        self.channel_page = 0
        self.focus_channel_idx = 0

        self.last_window = None
        self.marker_a = None
        self.marker_b = None

        self._build_ui()
        self.load_file(0)

    def _build_ui(self):
        self.fig = plt.figure(figsize=(20, 11))
        self.fig.canvas.manager.set_window_title("Clinical-style EEG Viewer")

        self.ax_file_list = self.fig.add_axes([0.02, 0.58, 0.16, 0.30])
        self.ax_file_slider = self.fig.add_axes([0.185, 0.58, 0.010, 0.30])

        self.ax_checks = self.fig.add_axes([0.02, 0.35, 0.10, 0.16])
        self.ax_info = self.fig.add_axes([0.12, 0.35, 0.08, 0.16])

        self.ax_stack = self.fig.add_axes([0.23, 0.50, 0.55, 0.40])
        self.ax_psd = self.fig.add_axes([0.23, 0.16, 0.55, 0.20])
        self.ax_metrics = self.fig.add_axes([0.80, 0.16, 0.18, 0.74])

        self.ax_metrics.axis("off")
        self.ax_metrics.set_title("Readout / Hints", fontsize=11)

        self.ax_time_slider = self.fig.add_axes([0.30, 0.08, 0.34, 0.03])

        ax_zoom_in = self.fig.add_axes([0.66, 0.06, 0.035, 0.045])
        ax_zoom_out = self.fig.add_axes([0.70, 0.06, 0.035, 0.045])
        ax_reprocess = self.fig.add_axes([0.02, 0.28, 0.08, 0.05])
        ax_ch_prev = self.fig.add_axes([0.11, 0.28, 0.04, 0.05])
        ax_ch_next = self.fig.add_axes([0.16, 0.28, 0.04, 0.05])
        ax_clear_marks = self.fig.add_axes([0.80, 0.08, 0.08, 0.05])
        ax_swap_marks = self.fig.add_axes([0.89, 0.08, 0.08, 0.05])

        self.btn_zoom_in = Button(ax_zoom_in, "+")
        self.btn_zoom_out = Button(ax_zoom_out, "-")
        self.btn_reprocess = Button(ax_reprocess, "Reprocess")
        self.btn_ch_prev = Button(ax_ch_prev, "Ch-")
        self.btn_ch_next = Button(ax_ch_next, "Ch+")
        self.btn_clear_marks = Button(ax_clear_marks, "Clear A/B")
        self.btn_swap_marks = Button(ax_swap_marks, "Swap A/B")

        self.btn_zoom_in.on_clicked(self.on_zoom_in)
        self.btn_zoom_out.on_clicked(self.on_zoom_out)
        self.btn_reprocess.on_clicked(self.on_reprocess)
        self.btn_ch_prev.on_clicked(self.on_ch_prev_page)
        self.btn_ch_next.on_clicked(self.on_ch_next_page)
        self.btn_clear_marks.on_clicked(self.on_clear_marks)
        self.btn_swap_marks.on_clicked(self.on_swap_marks)

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

        self.check = CheckButtons(
            self.ax_checks,
            ["Bandpass", "Notch50", "AvgRef", "InterpBad"],
            [self.enable_bandpass, self.enable_notch, self.enable_avg_ref, self.enable_interp]
        )
        self.ax_checks.set_title("Preprocess", fontsize=10)
        self.check.on_clicked(self.on_toggle_preprocess)

        self.ax_info.axis("off")

        self.time_slider = Slider(
            self.ax_time_slider, "Start (s)", 0, 1,
            valinit=0, valstep=0.1
        )
        self.time_slider.on_changed(self.on_time_slider_changed)

        self.fig.canvas.mpl_connect("button_press_event", self.on_stack_click)

    def _format_file_label(self, path):
        rel = path.relative_to(self.eeg_root)
        text = str(rel)
        return text if len(text) <= 28 else "..." + text[-25:]

    def _detect_bad_channels(self, raw, flat_thresh=1e-12, std_z_thresh=5.0, max_bad_ratio=0.3):
        bads = []
        try:
            eeg_raw = raw.copy().pick("eeg")
            if len(eeg_raw.ch_names) == 0:
                return bads
            data = eeg_raw.get_data()
            ch_names = eeg_raw.ch_names

            finite_mask = np.all(np.isfinite(data), axis=1)
            for i, ok in enumerate(finite_mask):
                if not ok:
                    bads.append(ch_names[i])

            valid_idx = np.where(finite_mask)[0]
            if len(valid_idx) == 0:
                return sorted(set(bads))

            valid_data = data[valid_idx]
            valid_names = [ch_names[i] for i in valid_idx]
            ch_std = np.std(valid_data, axis=1)

            flat_mask = ch_std < flat_thresh
            for i, is_flat in enumerate(flat_mask):
                if is_flat:
                    bads.append(valid_names[i])

            median_std = np.median(ch_std)
            mad_std = np.median(np.abs(ch_std - median_std)) + 1e-12
            robust_z = 0.6745 * (ch_std - median_std) / mad_std
            outlier_mask = np.abs(robust_z) > std_z_thresh
            for i, is_outlier in enumerate(outlier_mask):
                if is_outlier:
                    bads.append(valid_names[i])

            bads = list(sorted(set(bads)))
            max_bad_num = max(1, int(len(ch_names) * max_bad_ratio))
            if len(bads) > max_bad_num:
                return []
        except Exception:
            return []
        return bads

    def _preprocess_raw(self, raw):
        raw = raw.copy()
        try:
            raw.pick("eeg")
        except Exception:
            pass

        auto_bads = self._detect_bad_channels(raw)
        if auto_bads:
            raw.info["bads"] = list(sorted(set(raw.info.get("bads", []) + auto_bads)))

        if self.enable_interp and len(raw.info.get("bads", [])) > 0:
            try:
                montage = mne.channels.make_standard_montage("standard_1020")
                raw.set_montage(montage, on_missing="ignore", verbose=False)
                if np.all(np.isfinite(raw.get_data())):
                    raw.interpolate_bads(reset_bads=False, verbose=False)
            except Exception:
                pass

        if self.enable_avg_ref:
            try:
                raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
            except Exception:
                pass

        if self.enable_bandpass:
            try:
                raw.filter(self.l_freq, self.h_freq, fir_design="firwin",
                           skip_by_annotation="edge", verbose=False)
            except Exception:
                pass

        if self.enable_notch:
            try:
                raw.notch_filter(freqs=[self.notch_freq], fir_design="firwin", verbose=False)
            except Exception:
                pass

        return raw, auto_bads

    def load_file(self, index):
        self.file_index = index
        edf_path = self.edf_files[index]
        print(f"Loading: {edf_path}")

        self.raw_orig = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        self.raw_proc, self.auto_bads = self._preprocess_raw(self.raw_orig)

        self.channels = self.raw_proc.ch_names
        self.focus_channel_idx = 0
        self.channel_page = 0
        self.start_sec = 0.0
        self.marker_a = None
        self.marker_b = None

        self.file_list.set_selected(index)
        self._reset_time_slider()
        self.plot_all()

    def _reset_time_slider(self):
        total_duration = self.raw_proc.times[-1] if len(self.raw_proc.times) > 0 else 1
        max_start = max(0, total_duration - self.win_sec)
        self.time_slider.valmin = 0
        self.time_slider.valmax = max(1, max_start)
        self.time_slider.ax.set_xlim(self.time_slider.valmin, self.time_slider.valmax)
        self.time_slider.set_val(self.start_sec)

    def _visible_channel_indices(self):
        start = self.channel_page * self.channels_per_page
        stop = min(start + self.channels_per_page, len(self.channels))
        return list(range(start, stop))

    def on_file_selected(self, index):
        self.load_file(index)

    def on_file_slider_changed(self, val):
        self.file_list.set_start(int(val))

    def on_time_slider_changed(self, val):
        self.start_sec = float(val)
        self.plot_all()

    def on_zoom_in(self, event):
        self.win_sec = max(1.0, self.win_sec / 2.0)
        self._reset_time_slider()
        self.plot_all()

    def on_zoom_out(self, event):
        total_duration = self.raw_proc.times[-1] if len(self.raw_proc.times) > 0 else self.win_sec
        self.win_sec = min(total_duration, self.win_sec * 2.0)
        self._reset_time_slider()
        self.plot_all()

    def on_toggle_preprocess(self, label):
        if label == "Bandpass":
            self.enable_bandpass = not self.enable_bandpass
        elif label == "Notch50":
            self.enable_notch = not self.enable_notch
        elif label == "AvgRef":
            self.enable_avg_ref = not self.enable_avg_ref
        elif label == "InterpBad":
            self.enable_interp = not self.enable_interp

    def on_reprocess(self, event):
        self.raw_proc, self.auto_bads = self._preprocess_raw(self.raw_orig)
        self.channels = self.raw_proc.ch_names
        self.focus_channel_idx = min(self.focus_channel_idx, len(self.channels) - 1)
        max_page = max(0, (len(self.channels) - 1) // self.channels_per_page)
        self.channel_page = min(self.channel_page, max_page)
        self.plot_all()

    def on_ch_prev_page(self, event):
        self.channel_page = max(0, self.channel_page - 1)
        self.plot_all()

    def on_ch_next_page(self, event):
        max_page = max(0, (len(self.channels) - 1) // self.channels_per_page)
        self.channel_page = min(max_page, self.channel_page + 1)
        self.plot_all()

    def on_clear_marks(self, event):
        self.marker_a = None
        self.marker_b = None
        self.plot_all()

    def on_swap_marks(self, event):
        self.marker_a, self.marker_b = self.marker_b, self.marker_a
        self.plot_all()

    def _get_window_data(self):
        sfreq = float(self.raw_proc.info["sfreq"])
        start_sample = int(self.start_sec * sfreq)
        stop_sample = int(min((self.start_sec + self.win_sec) * sfreq, len(self.raw_proc.times)))

        visible_idx = self._visible_channel_indices()
        data_proc, times = self.raw_proc[visible_idx, start_sample:stop_sample]
        data_proc_uv = data_proc * 1e6

        orig_map = {ch: i for i, ch in enumerate(self.raw_orig.ch_names)}
        data_orig_uv = []
        for idx in visible_idx:
            ch = self.raw_proc.ch_names[idx]
            if ch in orig_map:
                d, _ = self.raw_orig[orig_map[ch], start_sample:stop_sample]
                data_orig_uv.append(d[0] * 1e6)
            else:
                data_orig_uv.append(np.zeros_like(times))
        data_orig_uv = np.array(data_orig_uv)

        return visible_idx, data_orig_uv, data_proc_uv, times, sfreq

    def _compute_band_powers(self, x, sfreq):
        x = np.asarray(x, dtype=float)
        if len(x) < 4:
            return None
        x = x - np.mean(x)
        n = len(x)
        freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
        fft_vals = np.fft.rfft(x)
        psd = (np.abs(fft_vals) ** 2) / max(n, 1)

        valid = freqs <= 60
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
            "beta": bandpower(13, 30),
            "gamma": bandpower(30, 45),
        }
        rel = {k: v / total_power for k, v in bands.items()}
        dominant_freq = freqs[np.argmax(psd)] if len(freqs) else 0.0
        return {
            "freqs": freqs,
            "psd": psd,
            "dominant_freq": float(dominant_freq),
            "rel_bands": rel,
        }

    def _find_spike_candidates(self, x, sfreq):
        x = np.asarray(x, dtype=float)
        if len(x) < 5:
            return np.array([], dtype=int)
        dx = np.diff(x, prepend=x[0])
        mad = np.median(np.abs(dx - np.median(dx))) + 1e-12
        zdx = np.abs(dx - np.median(dx)) / (1.4826 * mad)
        amp_thr = np.median(np.abs(x - np.median(x))) * 4.0 + 1e-12

        refractory = int(0.08 * sfreq)
        last_idx = -refractory
        out = []
        for i in range(1, len(x) - 1):
            local_peak = ((x[i] > x[i - 1] and x[i] > x[i + 1]) or
                          (x[i] < x[i - 1] and x[i] < x[i + 1]))
            if not local_peak:
                continue
            if zdx[i] < 5.0:
                continue
            if abs(x[i] - np.median(x)) < amp_thr:
                continue
            if i - last_idx < refractory:
                continue
            out.append(i)
            last_idx = i
        return np.array(out, dtype=int)

    def _estimate_clicked_channel_and_value(self, event):
        if self.last_window is None or event.inaxes != self.ax_stack or event.xdata is None or event.ydata is None:
            return None

        visible_idx = self.last_window["visible_idx"]
        offsets = self.last_window["offsets"]
        proc = self.last_window["data_proc_uv"]
        times = self.last_window["times"]

        t_click = float(event.xdata)
        y_click = float(event.ydata)

        nearest_time_idx = int(np.argmin(np.abs(times - t_click)))
        channel_dist = [abs(y_click - off) for off in offsets]
        row = int(np.argmin(channel_dist))
        ch_idx = visible_idx[row]
        amp = float(proc[row, nearest_time_idx])
        t = float(times[nearest_time_idx])
        y_plot = float(amp + offsets[row])

        return {
            "row": row,
            "channel_global_idx": ch_idx,
            "channel_name": self.channels[ch_idx],
            "time": t,
            "amp_uv": amp,
            "time_idx": nearest_time_idx,
            "y_plot": y_plot,
        }

    def on_stack_click(self, event):
        info = self._estimate_clicked_channel_and_value(event)
        if info is None:
            return

        if event.button == 1:
            self.marker_a = info
            self.focus_channel_idx = info["channel_global_idx"]
        elif event.button == 3:
            self.marker_b = info
            self.focus_channel_idx = info["channel_global_idx"]
        else:
            return

        self.plot_all()

    def _draw_stack(self, visible_idx, data_orig_uv, data_proc_uv, times, sfreq):
        self.ax_stack.clear()

        robust_scale = np.median(np.std(data_proc_uv, axis=1)) * 4.0
        if not np.isfinite(robust_scale) or robust_scale <= 1e-6:
            robust_scale = 50.0

        offsets = []
        for i, ch_idx in enumerate(visible_idx):
            offset = (len(visible_idx) - 1 - i) * robust_scale * 3.0
            offsets.append(offset)

            ch_name = self.channels[ch_idx]
            lw = 1.2 if ch_idx == self.focus_channel_idx else 0.8
            alpha = 1.0 if ch_idx == self.focus_channel_idx else 0.85

            self.ax_stack.plot(times, data_orig_uv[i] + offset, linewidth=0.5, alpha=0.20)
            self.ax_stack.plot(times, data_proc_uv[i] + offset, linewidth=lw, alpha=alpha)

            self.ax_stack.text(
                times[0] - 0.01 * self.win_sec,
                offset,
                ch_name,
                ha="right",
                va="center",
                fontsize=9
            )

            spikes = self._find_spike_candidates(data_proc_uv[i], sfreq)
            if len(spikes) > 0:
                self.ax_stack.scatter(
                    times[spikes],
                    data_proc_uv[i, spikes] + offset,
                    s=10
                )

        self.ax_stack.set_title(
            f"File: {self.edf_files[self.file_index].name}\n"
            f"Channels page: {self.channel_page + 1} | "
            f"Window: {self.start_sec:.2f}s - {self.start_sec + self.win_sec:.2f}s"
        )
        self.ax_stack.set_xlabel("Time (s)")
        self.ax_stack.set_yticks([])
        self.ax_stack.grid(True, alpha=0.25)

        for mk, label in [(self.marker_a, "A"), (self.marker_b, "B")]:
            if mk is not None and mk["channel_global_idx"] in visible_idx:
                self.ax_stack.axvline(mk["time"], linestyle="--", linewidth=0.9, alpha=0.8)
                self.ax_stack.text(
                    mk["time"], mk["y_plot"], label,
                    fontsize=10, ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
                )

        self.last_window = {
            "visible_idx": visible_idx,
            "data_proc_uv": data_proc_uv,
            "times": times,
            "offsets": offsets,
            "scale": robust_scale,
        }

    def _draw_psd(self, x, sfreq):
        self.ax_psd.clear()
        bp = self._compute_band_powers(x, sfreq)
        if bp is None:
            return None

        self.ax_psd.plot(bp["freqs"], bp["psd"], linewidth=1.0)
        self.ax_psd.set_xlim(0, 60)
        self.ax_psd.set_xlabel("Frequency (Hz)")
        self.ax_psd.set_ylabel("Power")
        self.ax_psd.set_title(f"Power Spectrum - focus channel: {self.channels[self.focus_channel_idx]}")
        self.ax_psd.grid(True, alpha=0.3)

        for f in [4, 8, 13, 30, 50]:
            self.ax_psd.axvline(f, linestyle="--", linewidth=0.8, alpha=0.5)
        return bp

    def _draw_right_panel(self, bp, data_proc_uv, visible_idx, sfreq):
        self.ax_metrics.clear()
        self.ax_metrics.axis("off")
        self.ax_metrics.set_title("Readout / Hints", fontsize=11)

        focus_row = visible_idx.index(self.focus_channel_idx) if self.focus_channel_idx in visible_idx else 0
        x = data_proc_uv[focus_row]
        ptp = float(np.ptp(x))
        std = float(np.std(x))
        rms = float(np.sqrt(np.mean(x ** 2)))
        ll = float(np.sum(np.abs(np.diff(x))) / max(len(x) - 1, 1))
        spikes = self._find_spike_candidates(x, sfreq)

        lines = [
            f"Focus ch:      {self.channels[self.focus_channel_idx]}",
            f"Bad channels:  {', '.join(self.auto_bads) if self.auto_bads else 'None'}",
            "",
            f"Bandpass:      {self.enable_bandpass} ({self.l_freq}-{self.h_freq} Hz)",
            f"Notch50:       {self.enable_notch}",
            f"AvgRef:        {self.enable_avg_ref}",
            f"InterpBad:     {self.enable_interp}",
            "",
            f"Peak-to-peak:  {ptp:.2f} uV",
            f"STD:           {std:.2f} uV",
            f"RMS:           {rms:.2f} uV",
            f"Line length:   {ll:.2f}",
            f"Spike cand.:   {len(spikes)}",
        ]

        if bp is not None:
            slow_ratio = bp["rel_bands"]["delta"] + bp["rel_bands"]["theta"]
            lines += [
                "",
                f"Dominant freq: {bp['dominant_freq']:.2f} Hz",
                f"Delta:         {bp['rel_bands']['delta']:.3f}",
                f"Theta:         {bp['rel_bands']['theta']:.3f}",
                f"Alpha:         {bp['rel_bands']['alpha']:.3f}",
                f"Beta:          {bp['rel_bands']['beta']:.3f}",
                f"Gamma:         {bp['rel_bands']['gamma']:.3f}",
                f"Slow ratio:    {slow_ratio:.3f}",
            ]

        lines += ["", "Click readout:"]
        if self.marker_a is not None:
            lines.append(
                f"A: {self.marker_a['channel_name']}, t={self.marker_a['time']:.4f}s, "
                f"amp={self.marker_a['amp_uv']:.2f}uV"
            )
        else:
            lines.append("A: not set (left click)")

        if self.marker_b is not None:
            lines.append(
                f"B: {self.marker_b['channel_name']}, t={self.marker_b['time']:.4f}s, "
                f"amp={self.marker_b['amp_uv']:.2f}uV"
            )
        else:
            lines.append("B: not set (right click)")

        if self.marker_a is not None and self.marker_b is not None:
            dt = self.marker_b["time"] - self.marker_a["time"]
            dv = self.marker_b["amp_uv"] - self.marker_a["amp_uv"]
            slope = dv / dt if abs(dt) > 1e-12 else np.nan
            lines += [
                "",
                f"dt:            {dt * 1000:.2f} ms",
                f"dv:            {dv:.2f} uV",
                f"slope:         {slope:.2f} uV/s",
            ]

        hints = []
        if bp is not None:
            slow_ratio = bp["rel_bands"]["delta"] + bp["rel_bands"]["theta"]
            if slow_ratio > 0.60:
                hints.append("Slow activity is prominent in the focus channel.")
            if bp["dominant_freq"] < 8.0:
                hints.append("Dominant frequency is relatively slow.")
            if self.enable_notch:
                hints.append("50 Hz notch is enabled; inspect PSD near 50 Hz.")

        if len(spikes) >= 3:
            hints.append("Several sharp transient candidates in focus channel.")

        lines += ["", "Clinical reading ideas:"]
        if hints:
            lines += [f"- {x}" for x in hints]
        else:
            lines += ["- No obvious heuristic alert in this window."]

        lines += [
            "",
            "Doctor usually looks for:",
            "- same-time changes across nearby channels",
            "- rhythmic evolution over seconds",
            "- spatial spread from one region",
            "- spike/sharp-wave morphology",
            "",
            "Left click = mark A",
            "Right click = mark B",
        ]

        self.ax_metrics.text(
            0.02, 0.98, "\n".join(lines),
            va="top", ha="left",
            fontsize=9, family="monospace"
        )

    def plot_all(self):
        self.ax_info.clear()
        self.ax_info.axis("off")
        self.ax_info.text(
            0.02, 0.95,
            f"Ch page\n{self.channel_page + 1}/{max(1, (len(self.channels)-1)//self.channels_per_page + 1)}\n\n"
            f"Visible\n{self.channels_per_page}\n\n"
            f"Focus\n{self.channels[self.focus_channel_idx] if self.channels else 'N/A'}",
            va="top", ha="left", fontsize=10
        )

        visible_idx, data_orig_uv, data_proc_uv, times, sfreq = self._get_window_data()
        self._draw_stack(visible_idx, data_orig_uv, data_proc_uv, times, sfreq)

        focus_row = visible_idx.index(self.focus_channel_idx) if self.focus_channel_idx in visible_idx else 0
        bp = self._draw_psd(data_proc_uv[focus_row], sfreq)
        self._draw_right_panel(bp, data_proc_uv, visible_idx, sfreq)

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == "__main__":
    eeg_folder = "../data/epilepsy_eeg"   # 改成你的目录
    viewer = MultiChannelEDFViewer(eeg_folder)
    viewer.show()
