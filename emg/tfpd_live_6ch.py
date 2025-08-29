import numpy as np
import matplotlib.pyplot as plt
import threading, time, struct, collections, csv, os
from queue import Queue, Empty
import serial
import pywt
from scipy.signal import butter, filtfilt, spectrogram, get_window

# ====================== USER SETTINGS ======================
SERIAL_PORT = 'COM6'      # <-- change if needed
BAUDRATE    = 115200
NUM_CH      = 6
PACKET_SIZE = 2 + (NUM_CH + 1) * 2  # header(2) + NUM_CH ADC + 1 timer (all uint16 LE)

fs      = 250        # Hz
winLen  = 62        # samples
overlap = 0          # samples
nfft    = int(2 ** np.ceil(np.log2(winLen)))

# wavelet denoise (not used for raw plot)
wvl   = "sym8"
level = 1
thres_method = "soft"
mode  = "per"

# bandpass (15–124 Hz) used only for TFPD
f1, f2 = 15, 124
order  = 4

RAW_SECONDS     = 10                 # show last N seconds of raw
RAW_BUF_SAMPLES = RAW_SECONDS * fs
UPDATE_MS       = 200                # plot update period (ms)

# stats print period
PRINT_INTERVAL  = 1.0

# ---- warm-up (ignore first N seconds after acquisition starts) ----
GRACE_SECONDS   = 1.0
acq_start_time  = None
grace_end_time  = None
grace_done      = False

# ---- Figure & Axes scaling (fixed) ----
FIG_SIZE   = (12, 2.8*6)   # inches (width, height) - scaled for 6 channels
FIG_DPI    = 100

# Raw axis scaling — set to None to auto
RAW_YLIM   = (0, 4095)      # or None for auto padding
RAW_YPAD   = 0.10           # only used when RAW_YLIM is None

# Raw X-axis behavior
# 'sliding'  -> always show the last RAW_SECONDS
# 'zero_based' -> fixed 0..RAW_SECONDS (doesn't move)
RAW_X_MODE = "sliding"

# TFPD axis scaling (normalized to [-1, 1]), keep fixed
TFPD_YLIM    = (-1.0, 1.0)
TFPD_X_FIXED = True
# ===========================================================

# --------------- Helpers ---------------
def wavelet_denoise(x, wavelet, level, method="soft"):
    if len(x) < 4:
        return x.astype(float, copy=True)
    coeffs = pywt.wavedec(x, wavelet, mode="per", level=level)
    sigma_est = np.median(np.abs(coeffs[-1]))/0.6745 if len(coeffs[-1]) else 0.0
    uthresh = sigma_est * np.sqrt(2*np.log(len(x))) if sigma_est > 0 else 0.0
    if uthresh > 0:
        coeffs[1:] = [pywt.threshold(c, value=uthresh, mode=method) for c in coeffs[1:]]
    y = pywt.waverec(coeffs, wavelet, mode="per")
    return y[:len(x)]

def design_bandpass(fs, f1, f2, order):
    nyq = fs/2
    if f2 >= nyq:
        f2 = max(f1+1, nyq-1)
    b, a = butter(order, [f1/nyq, f2/nyq], btype="band")
    return b, a

# --------------- Serial Reader ---------------
running = True
ser = None

# Per-channel buffers for live plotting/processing (after grace)
adc_bufs = [collections.deque([0]*RAW_BUF_SAMPLES, maxlen=RAW_BUF_SAMPLES) for _ in range(NUM_CH)]
buf_lock = threading.Lock()

# Per-channel GRACE buffers (ignored samples used to compute baseline σ)
GRACE_BUF_SAMPLES = max(1, int(GRACE_SECONDS * fs))
grace_bufs = [collections.deque(maxlen=GRACE_BUF_SAMPLES) for _ in range(NUM_CH)]
grace_lock = threading.Lock()

# ---- Streaming frame parameters & per-channel TFPD buffers ----
HOP_SAMPLES = max(1, winLen - overlap)     # STFT hop length
frame_dt    = HOP_SAMPLES / fs             # seconds per TFPD frame
TFPD_BUF_FRAMES = max(1, int(np.ceil(RAW_BUF_SAMPLES / HOP_SAMPLES)))
tfpd_bufs = [collections.deque(maxlen=TFPD_BUF_FRAMES) for _ in range(NUM_CH)]  # per-channel normalized TFPD history

# Counters to append only new frames (shared timing across channels)
samples_post_grace = 0   # total samples received after grace
frames_emitted     = 0   # frames appended so far (per shared stream)

def _frames_possible(sample_count):
    """How many frames can be formed from sample_count with (winLen, HOP_SAMPLES)."""
    if sample_count < winLen:
        return 0
    return 1 + (sample_count - winLen) // HOP_SAMPLES

# ---------------- Recording (channel selection) ----------------
RECORD_ENABLED_DEFAULT = False
RECORD_CHANNELS = [0, 1, 2]   # <-- edit this list or use number keys 1–6 to toggle (when not recording)
RECORD_DIR = "records"

record_enabled = RECORD_ENABLED_DEFAULT
record_channels = set([ch for ch in RECORD_CHANNELS if 0 <= ch < NUM_CH])
record_lock = threading.Lock()

rec_queue = Queue(maxsize=10000)
rec_writer_thread = None
rec_writer_stop = threading.Event()
rec_file_handle = None
rec_csv_writer = None
rec_filename = None

def _record_header_channels():
    with record_lock:
        cols = sorted(record_channels)
    return cols

def _make_filename():
    ts = time.strftime("%Y%m%d_%H%M%S")
    chs = "-".join([f"ch{c}" for c in _record_header_channels()])
    os.makedirs(RECORD_DIR, exist_ok=True)
    return os.path.join(RECORD_DIR, f"emg_record_{ts}_{chs}.csv")

def start_recording():
    global rec_writer_thread, rec_file_handle, rec_csv_writer, rec_filename, record_enabled
    with record_lock:
        if record_enabled:
            print("[REC] Already recording.")
            return
        if len(record_channels) == 0:
            print("[REC] No channels selected. Select channels before recording.")
            return
        record_enabled = True

    rec_writer_stop.clear()
    rec_filename = _make_filename()
    try:
        rec_file_handle = open(rec_filename, "w", newline="")
        rec_csv_writer = csv.writer(rec_file_handle)
        # header
        cols = _record_header_channels()
        header = ["t_sec"] + [f"ch{c}" for c in cols]
        rec_csv_writer.writerow(header)
        rec_file_handle.flush()
    except Exception as e:
        print(f"[REC] Failed to open file: {e}")
        with record_lock:
            record_enabled = False
        return

    rec_writer_thread = threading.Thread(target=_rec_writer_loop, daemon=True)
    rec_writer_thread.start()
    print(f"[REC] Started -> {rec_filename} | channels={sorted(record_channels)}")

def stop_recording():
    global rec_writer_thread, rec_file_handle, rec_csv_writer, rec_filename, record_enabled
    with record_lock:
        if not record_enabled:
            print("[REC] Not recording.")
            return
        record_enabled = False

    rec_writer_stop.set()
    if rec_writer_thread is not None:
        rec_writer_thread.join(timeout=2.0)
        rec_writer_thread = None

    try:
        if rec_file_handle is not None:
            rec_file_handle.flush()
            rec_file_handle.close()
    finally:
        rec_file_handle = None
        rec_csv_writer = None
        print(f"[REC] Stopped -> {rec_filename}")
        rec_filename = None

def _rec_writer_loop():
    # Drain queue until told to stop and queue is empty
    while not rec_writer_stop.is_set() or not rec_queue.empty():
        try:
            row = rec_queue.get(timeout=0.2)
        except Empty:
            continue
        try:
            if rec_csv_writer is not None:
                rec_csv_writer.writerow(row)
                if (rec_queue.qsize() % 2000) == 0:
                    # occasional flush
                    rec_file_handle.flush()
        except Exception as e:
            print(f"[REC] Write error: {e}")

def set_record_channels(ch_list):
    """Programmatic selection of channels to record (takes effect when NOT recording)."""
    global record_channels
    with record_lock:
        if record_enabled:
            print("[REC] Cannot change channels while recording. Stop first (press 'r').")
            return False
        chs = set()
        for c in ch_list:
            if isinstance(c, int) and 0 <= c < NUM_CH:
                chs.add(c)
        record_channels = chs
    print(f"[REC] Selected channels: {sorted(record_channels)}")
    return True

def toggle_record_channel(c):
    """Toggle a single channel (0-based)."""
    with record_lock:
        if record_enabled:
            print("[REC] Cannot change channels while recording. Stop first (press 'r').")
            return
        if c in record_channels:
            record_channels.remove(c)
        else:
            if 0 <= c < NUM_CH:
                record_channels.add(c)
    print(f"[REC] Selected channels: {sorted(record_channels)}")

# --------------- Serial Open ---------------
def open_serial():
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        print(f"[INFO] Opened {SERIAL_PORT}")
    except Exception as e:
        print(f"[ERROR] Could not open {SERIAL_PORT}: {e}")
        ser = None

def serial_reader():
    global running, acq_start_time, grace_done, grace_end_time, samples_post_grace
    while running:
        try:
            if ser is None:
                time.sleep(0.2)
                continue
            if ser.read(2) != b'\xAA\x55':
                continue
            payload = ser.read(PACKET_SIZE - 2)
            if len(payload) != PACKET_SIZE - 2:
                continue
            values = struct.unpack('<' + 'H'*NUM_CH + ' H', payload)  # NUM_CH ADCs + timer
            ch_vals = values[:NUM_CH]

            now = time.time()
            if acq_start_time is None:
                acq_start_time = now
                print(f"[INFO] Acquisition started. Ignoring first {GRACE_SECONDS:.1f}s...")

            # During grace: collect baseline samples, don't push to adc_bufs
            if (now - acq_start_time) < GRACE_SECONDS:
                with grace_lock:
                    for ch in range(NUM_CH):
                        grace_bufs[ch].append(ch_vals[ch])
                continue

            # First sample after grace:
            if not grace_done:
                with buf_lock:
                    for ch in range(NUM_CH):
                        adc_bufs[ch].clear()
                grace_done = True
                grace_end_time = now
                samples_post_grace = 0
                print("[INFO] Grace period over. Starting TFPD & live plotting for all channels.")

            # After grace, push to per-channel main buffers and count samples for frame pacing
            with buf_lock:
                for ch in range(NUM_CH):
                    adc_bufs[ch].append(ch_vals[ch])
            samples_post_grace += 1

            # Enqueue for recording if enabled
            with record_lock:
                rec_on = record_enabled
                cols = sorted(record_channels)
            if rec_on and grace_done:
                # Sample-based time (more stable than wall clock)
                t_sec = (samples_post_grace - 1) / fs
                row = [f"{t_sec:.6f}"]
                for c in cols:
                    row.append(str(ch_vals[c]))
                try:
                    rec_queue.put_nowait(row)
                except:
                    # If queue is full, drop silently to avoid blocking acquisition
                    pass

        except Exception:
            time.sleep(0.01)

# --------------- Processing + Plotting ---------------
b_bp, a_bp = design_bandpass(fs, f1, f2, order)

plt.ion()
fig, axs = plt.subplots(NUM_CH, 2, figsize=FIG_SIZE, dpi=FIG_DPI, constrained_layout=True)
if NUM_CH == 1:
    axs = np.array([[axs[0], axs[1]]])  # ensure 2D shape

ax_raw_list  = [axs[i,0] for i in range(NUM_CH)]
ax_tfpd_list = [axs[i,1] for i in range(NUM_CH)]

raw_lines = []
for ch in range(NUM_CH):
    ln, = ax_raw_list[ch].plot([], [], lw=1.0)
    raw_lines.append(ln)
    ax_raw_list[ch].set_title(f"CH{ch}: Raw (ADC)")
    ax_raw_list[ch].set_xlabel("Time (s)")
    ax_raw_list[ch].set_ylabel("ADC")
    ax_raw_list[ch].grid(True)

    ax_tfpd_list[ch].set_ylim(*TFPD_YLIM)
    if TFPD_X_FIXED:
        ax_tfpd_list[ch].set_xlim(0, RAW_SECONDS)
    ax_tfpd_list[ch].set_xlabel("Time (s) [STFT frame centers]")
    ax_tfpd_list[ch].set_ylabel("Norm. TFPD")
    ax_tfpd_list[ch].set_title(f"CH{ch}: Algorithm 2 Normalized TFPD")
    ax_tfpd_list[ch].grid(True)

# ---- Per-channel α/β (global min/max) & σ ----
alpha = [np.inf]*NUM_CH   # global_min per channel
beta  = [-np.inf]*NUM_CH  # global_max per channel
sigma = [np.nan]*NUM_CH
sigma_computed = [False]*NUM_CH

def _compute_sigma_from_grace(ch):
    """
    Compute baseline sigma from the latter half of the STFT frames
    of the GRACE (ignored) buffer for channel ch. Returns (float or None).
    """
    with grace_lock:
        g = np.array(grace_bufs[ch], dtype=np.float64)

    if g.size < max(winLen, 16):
        return None

    # Denoise + bandpass same as TFPD path
    g_denoised = wavelet_denoise(g, wvl, level, method=thres_method)
    try:
        g_filt = filtfilt(b_bp, a_bp, g_denoised)
    except Exception:
        g_filt = g_denoised

    fg, ttg, Sg = spectrogram(
        g_filt,
        fs=fs,
        window=get_window("hamming", winLen),
        nperseg=winLen,
        noverlap=overlap,
        nfft=nfft,
        detrend=False,
        scaling="density",
        mode="complex"
    )

    num_frames_g = Sg.shape[1]
    if num_frames_g == 0:
        return None

    # Use latter half frames
    start_idx = num_frames_g // 2
    baseline_idx = list(range(start_idx, num_frames_g)) if (num_frames_g - start_idx) > 0 else list(range(num_frames_g))
    P_baseline = np.abs(Sg[:, baseline_idx])**2
    if P_baseline.size == 0:
        return None

    return float(np.mean(P_baseline)) / 2.0

def process_current_buffer(ch):
    """
    Per-channel processing.
    Returns:
        t_raw, raw_for_plot, Normalized_TFPD (np.ndarrays)
    """
    # Get buffer snapshot
    with buf_lock:
        buf = np.array(adc_bufs[ch], dtype=np.float64)

    if buf.size < max(winLen, 16):
        return np.arange(buf.size)/fs, buf, np.array([])

    # RAW (unfiltered) for plot
    raw_unscaled = buf.copy()

    # Denoise (TFPD path only)
    x_denoised = wavelet_denoise(raw_unscaled, wvl, level, method=thres_method)

    # Bandpass
    try:
        filt = filtfilt(b_bp, a_bp, x_denoised)
    except Exception:
        filt = x_denoised.copy()

    f, tt, S = spectrogram(
        filt,
        fs=fs,
        window=get_window("hamming", winLen),
        nperseg=winLen,
        noverlap=overlap,
        nfft=nfft,
        detrend=False,
        scaling="density",
        mode="complex"
    )
    num_timeFrames = S.shape[1]

    # Compute sigma (once) from grace latter-half, else fallback to latter half of current
    if (not sigma_computed[ch]) and grace_done:
        s_cand = _compute_sigma_from_grace(ch)
        if s_cand is not None and np.isfinite(s_cand):
            sigma[ch] = s_cand
            sigma_computed[ch] = True
        else:
            if num_timeFrames > 0:
                start_idx = num_timeFrames // 2
                baseline_idx = list(range(start_idx, num_timeFrames)) if (num_timeFrames - start_idx) > 0 else list(range(num_timeFrames))
                P_baseline = np.abs(S[:, baseline_idx])**2
                sigma[ch] = float(np.mean(P_baseline)) if P_baseline.size > 0 else 0.0
                sigma_computed[ch] = True

    # TFPD for current window
    TFPD_time = np.zeros(num_timeFrames)
    if num_timeFrames > 0 and np.isfinite(sigma[ch]) and sigma[ch] > 0:
        m = winLen / fs
        n = 109
        for j in range(num_timeFrames):
            P = np.abs(S[:, j])**2
            recordF = f[P > sigma[ch]]
            TFPD_time[j] = len(recordF) / (m * n)

        # Update per-channel alpha/beta
        local_min = float(np.min(TFPD_time))
        local_max = float(np.max(TFPD_time))
        if local_min < alpha[ch]:
            alpha[ch] = local_min
        if local_max > beta[ch]:
            beta[ch] = local_max

    # Algorithm 2 normalization (per-channel α/β)
    Normalized_TFPD = np.full(num_timeFrames, -1.0)
    if num_timeFrames > 0 and np.isfinite(alpha[ch]) and np.isfinite(beta[ch]) and (beta[ch] > alpha[ch]):
        for j in range(num_timeFrames):
            if j < 2:
                Normalized_TFPD[j] = -1
            else:
                Normalized_TFPD[j] = ((2*TFPD_time[j] - alpha[ch]) / (beta[ch] - alpha[ch])) - 1

    t_raw = np.arange(raw_unscaled.size)/fs
    return t_raw, raw_unscaled, Normalized_TFPD

def redraw():
    global frames_emitted

    # Compute how many new frames can be appended this cycle
    frames_possible_now = _frames_possible(samples_post_grace)
    new_frames = max(0, frames_possible_now - frames_emitted)

    for ch in range(NUM_CH):
        t_raw, raw_for_plot, norm = process_current_buffer(ch)

        # ---- RAW plot per channel ----
        if t_raw.size > 0:
            raw_lines[ch].set_data(t_raw, raw_for_plot)

            # X limits
            if RAW_X_MODE == "sliding":
                ax_raw_list[ch].set_xlim(max(0, t_raw[-1]-RAW_SECONDS), max(RAW_SECONDS, t_raw[-1]))
            else:
                ax_raw_list[ch].set_xlim(0, RAW_SECONDS)

            # Y limits
            if RAW_YLIM is not None:
                ax_raw_list[ch].set_ylim(*RAW_YLIM)
            else:
                ymin, ymax = raw_for_plot.min(), raw_for_plot.max()
                if ymin == ymax:
                    ymin -= 1.0; ymax += 1.0
                yr = ymax - ymin
                ax_raw_list[ch].set_ylim(ymin - RAW_YPAD*yr, ymax + RAW_YPAD*yr)

        # ---- Append ONLY NEW frames into this channel's tfpd buffer ----
        if norm.size > 0 and new_frames > 0:
            k = min(new_frames, norm.size)
            to_add = norm[-k:]
            for v in to_add:
                tfpd_bufs[ch].append(float(v))

        # ---- TFPD plot (left → right fill) ----
        ax = ax_tfpd_list[ch]
        ax.cla()
        ax.set_ylim(*TFPD_YLIM)
        if TFPD_X_FIXED:
            ax.set_xlim(0, RAW_SECONDS)
        ax.set_xlabel("Time (s) [STFT frame centers]")
        ax.set_ylabel("Norm. TFPD")
        ax.set_title(f"CH{ch}: Algorithm 2 Normalized TFPD")
        ax.grid(True)

        if len(tfpd_bufs[ch]) > 0:
            vals = np.array(tfpd_bufs[ch], dtype=float)
            N = len(vals)
            tt_hist = (np.arange(N) + 0.5) * frame_dt  # left-anchored timeline
            ax.bar(tt_hist, vals, width=frame_dt)

    if new_frames > 0:
        frames_emitted += new_frames

    fig.canvas.draw_idle()

# --------------- Keyboard controls ---------------
def on_key(event):
    """Key bindings:
       1..6 : toggle record channel (when NOT recording)
       r    : start/stop recording
       s    : show current record channel selection
    """
    if event.key is None:
        return
    k = event.key.lower()
    if k in [str(i) for i in range(1, min(NUM_CH, 9)+1)]:
        ch = int(k) - 1
        toggle_record_channel(ch)
    elif k == 'r':
        with record_lock:
            rec_on = record_enabled
        if rec_on:
            stop_recording()
        else:
            start_recording()
    elif k == 's':
        with record_lock:
            print(f"[REC] Selected channels: {sorted(record_channels)}  | recording={record_enabled}")
    # else: ignore

# --------------- Main ---------------
def _fmt(v):
    return f"{v:.6g}" if np.isfinite(v) else "None"

def main():
    global running
    set_record_channels(RECORD_CHANNELS)  # validate initial selection
    open_serial()

    reader = threading.Thread(target=serial_reader, daemon=True)
    reader.start()

    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    print("[INFO] Key controls: 1–6 toggle channels (when not recording) | 'r' start/stop record | 's' show selection")

    try:
        last_plot = 0.0
        last_stat = 0.0
        while True:
            now = time.time()
            if grace_done and (now - last_plot) >= (UPDATE_MS/1000.0):
                redraw()
                last_plot = now

            if now - last_stat >= PRINT_INTERVAL:
                if grace_done:
                    parts = []
                    for ch in range(NUM_CH):
                        parts.append(f"CH{ch}:σ={_fmt(sigma[ch])},α={_fmt(alpha[ch])},β={_fmt(beta[ch])},N={len(tfpd_bufs[ch])}")
                    with record_lock:
                        rec_on = record_enabled
                        sel = sorted(record_channels)
                    print("[STATS] " + " | ".join(parts) + f" || REC={rec_on} ch={sel} q={rec_queue.qsize()}")
                else:
                    if acq_start_time is None:
                        print("[INFO] Waiting for first packet...")
                    else:
                        rem = max(0.0, GRACE_SECONDS - (now - acq_start_time))
                        print(f"[INFO] Warming up... {rem:.2f}s remaining")
                last_stat = now

            plt.pause(0.001)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        running = False
        stop_recording()
        if ser is not None:
            try: ser.close()
            except: pass
        time.sleep(0.1)

if __name__ == "__main__":
    main()
