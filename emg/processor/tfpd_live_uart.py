'''     
-----------------------------------------------------------------------------------------------
Code Snippet for controlling robot hand via STM32 microcontroller
Steps:  - Receiving EMG signal from STM32 ADC module via serial communicaion.
        - Signal filtering and denoising.
        - Calculate and normalize TFPD (Get control command).
        - Send control command to another STM32 via another serial port.    
-----------------------------------------------------------------------------------------------
'''

import numpy as np
import threading, time, struct, collections
import serial
import pywt
from scipy.signal import butter, filtfilt, spectrogram, get_window

# ====================== USER SETTINGS ======================
# Input (read) port
SERIAL_PORT_IN  = 'COM6'      # <-- change if needed
BAUDRATE_IN     = 11520

# Output (write) port for results
SERIAL_PORT_OUT = 'COM5'      # <-- send result out here
BAUDRATE_OUT    = 115200

NUM_CH      = 6
PACKET_SIZE = 2 + (NUM_CH + 1) * 2  # header(2) + 6 ADC + 1 timer (all uint16 LE)

fs      = 250        # Hz
winLen  = 62         # samples
overlap = 0          # samples
nfft    = int(2 ** np.ceil(np.log2(winLen)))

# wavelet denoise (used in TFPD path, not on raw)
wvl   = "sym8"
level = 1
thres_method = "soft"
mode  = "per"

# bandpass (15–124 Hz) used only for TFPD
f1, f2 = 15, 124
order  = 4

RAW_SECONDS     = 10                 # buffer span (s)
RAW_BUF_SAMPLES = RAW_SECONDS * fs

# console print periods
PRINT_INTERVAL  = 1.0     # stats print (s)
FRAME_PRINT     = True    # print per new frame

# ---- warm-up (ignore first N seconds after acquisition starts) ----
GRACE_SECONDS   = 1.0
acq_start_time  = None
grace_done      = False

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

# --------------- Serial Handles ---------------
running = True
ser_in  = None   # COM6 for input
ser_out = None   # COM5 for output

# Buffer for live processing (after grace)
adc_buf = collections.deque([0]*RAW_BUF_SAMPLES, maxlen=RAW_BUF_SAMPLES)
buf_lock = threading.Lock()

# Buffer for GRACE window (ignored samples used to compute baseline σ)
GRACE_BUF_SAMPLES = max(1, int(GRACE_SECONDS * fs))
grace_buf = collections.deque(maxlen=GRACE_BUF_SAMPLES)
grace_lock = threading.Lock()

# ---- Streaming frame parameters & bookkeeping ----
HOP_SAMPLES = max(1, winLen - overlap)     # STFT hop length
frame_dt    = HOP_SAMPLES / fs             # seconds per TFPD frame

# Counters to print only new frames
samples_post_grace = 0   # total samples received after grace
frames_emitted     = 0   # frames printed so far

def _frames_possible(sample_count):
    """How many frames can be formed from sample_count with (winLen, HOP_SAMPLES)."""
    if sample_count < winLen:
        return 0
    return 1 + (sample_count - winLen) // HOP_SAMPLES

def open_serial_in():
    global ser_in
    try:
        ser_in = serial.Serial(SERIAL_PORT_IN, BAUDRATE_IN, timeout=1)
        print(f"[INFO] Opened IN  {SERIAL_PORT_IN} @ {BAUDRATE_IN}")
    except Exception as e:
        print(f"[ERROR] Could not open input port {SERIAL_PORT_IN}: {e}")
        ser_in = None

def open_serial_out():
    global ser_out
    try:
        ser_out = serial.Serial(SERIAL_PORT_OUT, BAUDRATE_OUT, timeout=1)
        print(f"[INFO] Opened OUT {SERIAL_PORT_OUT} @ {BAUDRATE_OUT}")
    except Exception as e:
        print(f"[WARN] Could not open output port {SERIAL_PORT_OUT}: {e}")
        ser_out = None

def serial_reader():
    global running, acq_start_time, grace_done, samples_post_grace
    while running:
        try:
            if ser_in is None:
                time.sleep(0.2)
                continue
            if ser_in.read(2) != b'\xAA\x55':
                continue
            payload = ser_in.read(PACKET_SIZE - 2)
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
                print("[INFO] Grace period over. Starting TFPD processing.")

            # After grace, push to main buffer and count samples for frame pacing
            with buf_lock:
                adc_buf.append(ch0)
            samples_post_grace += 1

        except Exception:
            time.sleep(0.01)

# --------------- Processing ---------------
b_bp, a_bp = design_bandpass(fs, f1, f2, order)

# ---- GLOBAL alpha/beta across the whole session ----
global_min = np.inf   # alpha
global_max = -np.inf  # beta

# ---- current sigma for console print ----
sigma = np.nan
sigma_computed = False

# ---- For consecutive detection ----
prev_result = 0

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
        tfpd_vals (np.ndarray): raw (non-normalized) TFPD for current spectrogram frames
        norm_vals (np.ndarray): normalized TFPD for the same frames (Algorithm 2)
    """
    global global_min, global_max, sigma, sigma_computed

    with buf_lock:
        buf = np.array(adc_buf, dtype=np.float64)

    if buf.size < max(winLen, 16):
        return np.array([]), np.array([])

    # ---- Apply wavelet denoise for the TFPD path ----
    x_denoised = wavelet_denoise(buf, wvl, level, method=thres_method)

    # Bandpass filter (after denoise) for TFPD calculation
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
    if num_timeFrames == 0:
        return np.array([]), np.array([])

    # --- Ensure sigma is computed from GRACE latter-half frames (once) ---
    if not sigma_computed and grace_done:
        sigma_candidate = _compute_sigma_from_grace()
        if sigma_candidate is not None and np.isfinite(sigma_candidate):
            sigma = sigma_candidate
        else:
            start_idx = num_timeFrames // 2
            baseline_idx = list(range(start_idx, num_timeFrames)) if (num_timeFrames - start_idx) > 0 else list(range(num_timeFrames))
            P_baseline = np.abs(S[:, baseline_idx])**2
            sigma = float(np.mean(P_baseline)) / 2 if P_baseline.size > 0 else 0.0
        globals()['sigma_computed'] = True

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

    # Algorithm 2 normalization using global alpha/beta
    Normalized_TFPD = np.full(num_timeFrames, -1.0)
    if (num_timeFrames > 0 and
        np.isfinite(global_min) and np.isfinite(global_max) and
        (global_max > global_min)):
        for j in range(num_timeFrames):
            if j < 2:
                Normalized_TFPD[j] = -1
            else:
                Normalized_TFPD[j] = ((2*TFPD_time[j] - global_min) / (global_max - global_min)) - 1

    return TFPD_time, Normalized_TFPD

def _send_result_out(result_int: int):
    """Send result as ASCII '0\\n' or '1\\n' to COM5."""
    if ser_out is None:
        return
    try:
        ser_out.write(f"{int(result_int)}\n".encode('ascii'))
        print(f"[INFO] Sent: {int(result_int)}\n")
        ser_out.flush()
    except Exception as e:
        print(f"[UART ERROR] {e}")
        # Don’t spam; just note once every so often if needed
        # (Here we keep it silent after first failure to avoid console flood)
        pass

def process_and_print():
    """Emit console lines for any new frames that became available and send result to COM5."""
    global frames_emitted, prev_result

    tfpd_vals, norm_vals = process_current_buffer()
    if norm_vals.size == 0:
        return

    frames_possible_now = _frames_possible(samples_post_grace)
    new_frames = max(0, frames_possible_now - frames_emitted)
    if new_frames <= 0:
        return

    k = min(new_frames, norm_vals.size)
    base_idx = frames_emitted

    for i in range(k):
        frame_idx = base_idx + i
        t_center = (frame_idx + 0.5) * frame_dt
        norm_val = float(norm_vals[-k + i])

        tmp = 1 if norm_val > 0 else 0
        result = 1 if (prev_result == 1 and tmp == 1) else 0
        prev_result = tmp

        # ---- send result out via COM5 ----
        _send_result_out(result)

        if FRAME_PRINT:
            print(f"t={t_center:.3f}s | result={result}")

    frames_emitted += new_frames

# --------------- Main ---------------
def _fmt(v):
    return f"{v:.6g}" if np.isfinite(v) else "None"

def main():
    global running
    open_serial_in()
    open_serial_out()

    reader = threading.Thread(target=serial_reader, daemon=True)
    reader.start()
    try:
        last_stat = 0.0
        while True:
            now = time.time()

            if grace_done:
                process_and_print()

            if now - last_stat >= PRINT_INTERVAL:
                if grace_done:
                    # Optional stats:
                    # print(f"[STATS] sigma={_fmt(sigma)} | alpha={_fmt(global_min)} | beta={_fmt(global_max)} | frames_emitted={frames_emitted}")
                    pass
                else:
                    if acq_start_time is None:
                        print("[INFO] Waiting for first packet...")
                    else:
                        rem = max(0.0, GRACE_SECONDS - (now - acq_start_time))
                        print(f"[INFO] Warming up... {rem:.2f}s remaining")
                last_stat = now

            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        running = False
        if ser_in is not None:
            try: ser_in.close()
            except: pass
        if ser_out is not None:
            try: ser_out.close()
            except: pass
        time.sleep(0.1)

if __name__ == "__main__":
    main()
