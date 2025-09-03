'''     
-----------------------------------------------------------------------------------------------
Code Snippet for plotting live TFPD calculations for one EMG channel.     
-----------------------------------------------------------------------------------------------
'''
import numpy as np
import matplotlib.pyplot as plt
import threading, time, struct, collections
import serial
import pywt
from scipy.signal import butter, filtfilt, spectrogram, get_window

# ====================== USER SETTINGS ======================
SERIAL_PORT = 'COM6'      # <-- change if needed
BAUDRATE    = 115200
NUM_CH      = 6
PACKET_SIZE = 2 + (NUM_CH + 1) * 2  # header(2) + 6 ADC + 1 timer (all uint16 LE)

fs      = 250        # Hz
winLen  = 62       # samples
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
grace_done      = False

# ---- Figure & Axes scaling (fixed) ----
FIG_SIZE   = (11, 7)   # inches
FIG_DPI    = 100

# Raw (top) axis scaling — set to None to auto
# For a 12-bit ADC, 0..4095 is typical. Adjust if you use another range or offset.
RAW_YLIM   = (0, 4095)      # or None for auto padding
RAW_YPAD   = 0.10           # only used when RAW_YLIM is None

# X-axis behavior for raw plot
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

# --------------- Serial Reader (CH0 only) ---------------
running = True
ser = None

# Buffer for live plotting/processing (after grace)
adc_buf = collections.deque([0]*RAW_BUF_SAMPLES, maxlen=RAW_BUF_SAMPLES)
buf_lock = threading.Lock()

# Buffer for GRACE window (ignored samples used to compute baseline σ)
GRACE_BUF_SAMPLES = max(1, int(GRACE_SECONDS * fs))
grace_buf = collections.deque(maxlen=GRACE_BUF_SAMPLES)
grace_lock = threading.Lock()

# ---- Streaming frame parameters & TFPD buffer (for bottom plot) ----
HOP_SAMPLES = max(1, winLen - overlap)     # STFT hop length
frame_dt    = HOP_SAMPLES / fs             # seconds per TFPD frame
TFPD_BUF_FRAMES = max(1, int(np.ceil(RAW_BUF_SAMPLES / HOP_SAMPLES)))
tfpd_buf = collections.deque(maxlen=TFPD_BUF_FRAMES)  # stores normalized TFPD history

# Counters to append only new frames to tfpd_buf
samples_post_grace = 0   # total samples received after grace
frames_emitted     = 0   # frames appended so far

def _frames_possible(sample_count):
    """How many frames can be formed from sample_count with (winLen, HOP_SAMPLES)."""
    if sample_count < winLen:
        return 0
    return 1 + (sample_count - winLen) // HOP_SAMPLES

def open_serial():
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        print(f"[INFO] Opened {SERIAL_PORT}")
    except Exception as e:
        print(f"[ERROR] Could not open {SERIAL_PORT}: {e}")
        ser = None

def serial_reader():
    global running, acq_start_time, grace_done, samples_post_grace
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
            values = struct.unpack('<6H H', payload)
            ch0 = values[0]  # ADC counts (unscaled)

            now = time.time()
            if acq_start_time is None:
                acq_start_time = now
                print(f"[INFO] Acquisition started. Ignoring first {GRACE_SECONDS:.1f}s...")

            # During grace: collect baseline samples, don't push to adc_buf
            if (now - acq_start_time) < GRACE_SECONDS:
                with grace_lock:
                    grace_buf.append(ch0)
                continue

            # First sample after grace:
            if not grace_done:
                with buf_lock:
                    adc_buf.clear()
                grace_done = True
                samples_post_grace = 0
                print("[INFO] Grace period over. Starting TFPD & live plotting.")

            # After grace, push to main buffer and count samples for frame pacing
            with buf_lock:
                adc_buf.append(ch0)
            samples_post_grace += 1

        except Exception:
            time.sleep(0.01)

# --------------- Processing + Plotting ---------------
b_bp, a_bp = design_bandpass(fs, f1, f2, order)

plt.ion()
fig = plt.figure(constrained_layout=True, figsize=FIG_SIZE, dpi=FIG_DPI)
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
ax_raw = fig.add_subplot(gs[0, 0])
ax_tfpd = fig.add_subplot(gs[1, 0])

raw_line, = ax_raw.plot([], [], lw=1.0)
ax_raw.set_title("CH0: Raw (unscaled, unfiltered ADC values)")
ax_raw.set_xlabel("Time (s)")
ax_raw.set_ylabel("ADC counts")
ax_raw.grid(True)

ax_tfpd.set_ylim(*TFPD_YLIM)
ax_tfpd.set_xlabel("Time (s) [STFT frame centers]")
ax_tfpd.set_ylabel("Normalized TFPD")
ax_tfpd.set_title("Algorithm 2 Normalized TFPD (CH0)")
ax_tfpd.grid(True)

# ---- GLOBAL alpha/beta across the whole session ----
global_min = np.inf   # alpha
global_max = -np.inf  # beta

# ---- current sigma for console print ----
sigma = np.nan
sigma_computed = False

def _compute_sigma_from_grace():
    """
    Compute baseline sigma from the latter half of the STFT frames
    of the GRACE (ignored) buffer. Returns (sigma_value or None).
    """
    with grace_lock:
        g = np.array(grace_buf, dtype=np.float64)

    if g.size < max(winLen, 16):
        return None

    # Denoise + bandpass same as TFPD path (keeps consistency)
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

    # --- Use the latter half of the grace frames ---
    start_idx = num_frames_g // 2
    baseline_idx = list(range(start_idx, num_frames_g)) if (num_frames_g - start_idx) > 0 else list(range(num_frames_g))
    P_baseline = np.abs(Sg[:, baseline_idx])**2
    if P_baseline.size == 0:
        return None

    return float(np.mean(P_baseline)) / 2.0

def process_current_buffer():
    """
    Returns:
        t_raw (np.ndarray): time axis for raw (seconds)
        raw_for_plot (np.ndarray): RAW unfiltered ADC values for the top plot
        norm (np.ndarray): normalized TFPD values for the CURRENT adc_buf spectrogram
    """
    global global_min, global_max, sigma, sigma_computed

    with buf_lock:
        buf = np.array(adc_buf, dtype=np.float64)

    if buf.size < max(winLen, 16):
        return np.arange(buf.size)/fs, buf, np.array([])

    # RAW (unscaled) for top plot (truly raw)
    raw_unscaled = buf.copy()

    # ---- Apply wavelet denoise for the TFPD path (not for raw plot) ----
    x_denoised = wavelet_denoise(raw_unscaled, wvl, level, method=thres_method)

    # Bandpass filter (after denoise) for TFPD calculation
    try:
        filt = filtfilt(b_bp, a_bp, x_denoised)
    except Exception:
        # If filtfilt fails due to short buffer, just use the (denoised) signal
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

    if num_timeFrames == 0:
        return np.arange(raw_unscaled.size)/fs, raw_unscaled, np.array([])

    # --- Ensure sigma is computed from GRACE latter-half frames (once) ---
    if not sigma_computed and grace_done:
        sigma_candidate = _compute_sigma_from_grace()
        if sigma_candidate is not None and np.isfinite(sigma_candidate):
            sigma = sigma_candidate
            sigma_computed = True
        else:
            # Fallback: if grace insufficient, use latter half of current S frames
            start_idx = num_timeFrames // 2
            baseline_idx = list(range(start_idx, num_timeFrames)) if (num_timeFrames - start_idx) > 0 else list(range(num_timeFrames))
            P_baseline = np.abs(S[:, baseline_idx])**2
            sigma = float(np.mean(P_baseline)) / 2.0 if P_baseline.size > 0 else 0.0
            sigma_computed = True

    # --- Compute TFPD for current buffer using sigma ---
    m = winLen / fs
    n = 109
    TFPD_time = np.zeros(num_timeFrames)
    if np.isfinite(sigma) and sigma > 0:
        for j in range(num_timeFrames):
            P = np.abs(S[:, j])**2
            recordF = f[P > sigma]
            TFPD_time[j] = len(recordF) / (m * n)

    # update global alpha/beta across the whole session
    if TFPD_time.size > 0:
        local_min = float(np.min(TFPD_time))
        local_max = float(np.max(TFPD_time))
        if local_min < global_min:
            global_min = local_min
        if local_max > global_max:
            global_max = local_max

    # Algorithm 2 normalization using global alpha/beta; only compute on current window frames
    Normalized_TFPD = np.full(num_timeFrames, -1.0)
    if (num_timeFrames > 0 and
        np.isfinite(global_min) and np.isfinite(global_max) and
        (global_max > global_min)):
        for j in range(num_timeFrames):
            if j < 2:
                Normalized_TFPD[j] = -1
            else:
                Normalized_TFPD[j] = ((2*TFPD_time[j] - global_min) / (global_max - global_min)) - 1

    t_raw = np.arange(raw_unscaled.size)/fs
    return t_raw, x_denoised, Normalized_TFPD

def redraw():
    global frames_emitted

    t_raw, raw_for_plot, norm = process_current_buffer()

    # ---- RAW top plot ----
    if t_raw.size > 0:
        raw_line.set_data(t_raw, raw_for_plot)

        # X limits
        if RAW_X_MODE == "sliding":
            ax_raw.set_xlim(max(0, t_raw[-1]-RAW_SECONDS), max(RAW_SECONDS, t_raw[-1]))
        else:
            ax_raw.set_xlim(0, RAW_SECONDS)

        # Y limits
        if RAW_YLIM is not None:
            ax_raw.set_ylim(*RAW_YLIM)
        else:
            ymin, ymax = raw_for_plot.min(), raw_for_plot.max()
            if ymin == ymax:
                ymin -= 1.0; ymax += 1.0
            yr = ymax - ymin
            ax_raw.set_ylim(ymin - RAW_YPAD*yr, ymax + RAW_YPAD*yr)

    # ---- Append ONLY NEW frames into tfpd_buf ----
    if norm.size > 0:
        frames_possible_now = _frames_possible(samples_post_grace)
        new_frames = max(0, frames_possible_now - frames_emitted)
        if new_frames > 0:
            k = min(new_frames, norm.size)    # most-recent frames available now
            to_add = norm[-k:]
            for v in to_add:
                tfpd_buf.append(float(v))
            frames_emitted += new_frames

    # ---- TFPD bottom plot (LEFT → RIGHT) ----
    ax_tfpd.cla()
    ax_tfpd.set_ylim(*TFPD_YLIM)
    if TFPD_X_FIXED:
        ax_tfpd.set_xlim(0, RAW_SECONDS)
    ax_tfpd.set_xlabel("Time (s) [STFT frame centers]")
    ax_tfpd.set_ylabel("Normalized TFPD")
    ax_tfpd.set_title("Algorithm 2 Normalized TFPD (CH0)")
    ax_tfpd.grid(True)

    if len(tfpd_buf) > 0:
        vals = np.array(tfpd_buf, dtype=float)
        N = len(vals)
        # LEFT-anchored timeline: start at x=0 and fill to the right
        tt_hist = (np.arange(N) + 0.5) * frame_dt
        ax_tfpd.bar(tt_hist, vals, width=frame_dt)

    fig.canvas.draw_idle()

# --------------- Main ---------------
def _fmt(v):
    return f"{v:.6g}" if np.isfinite(v) else "None"

def main():
    global running
    open_serial()
    reader = threading.Thread(target=serial_reader, daemon=True)
    reader.start()
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
                    print(f"[STATS] sigma={_fmt(sigma)} | alpha={_fmt(global_min)} | beta={_fmt(global_max)} | tfpd_frames={len(tfpd_buf)}/{TFPD_BUF_FRAMES}")
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
        if ser is not None:
            try: ser.close()
            except: pass
        time.sleep(0.1)

if __name__ == "__main__":
    main()
