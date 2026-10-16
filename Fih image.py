"""
Basic acoustic image link on a 57 kHz channel.
TX: NI USB-6341 AO0 -> bench amp -> TX transducer
RX: RX transducer -> NI USB-6341 AI0
Sync: AI0 acquisition is started by the AO start trigger inside the 6341

Modulation: non-coherent BFSK, continuous phase, with two alternating tone
pairs (even symbols use pair A, odd use pair B) so the reverb tail of one
symbol doesn't land on the tones used by the next.

Frame: [LFM chirp][gap][8 training syms][16b frame idx][16b nbytes][payload][CRC16][tail]
Each frame is played once as a finite AO generation while AI0 records it,
then demodulated, CRC-checked, and resent on failure.
"""

import os
import time
import binascii
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, sosfiltfilt, fftconvolve

# ---------------- user settings ----------------
DEV         = "Dev1"            # NI USB-6341
AO_CHAN     = f"{DEV}/ao0"      # TX waveform -> amp input
AI_CHAN     = f"{DEV}/ai0"      # RX transducer
AI_TERM     = TerminalConfiguration.DIFF   # AI0+ / AI0- (pins 1 / 2)

IMAGE_IN    = "test.png"
IMG_W, IMG_H = 128, 128        # image is resized to this, 8-bit grayscale
IMAGE_OUT   = "rx_image.png"
COMPARE_OUT = "tx_vs_rx.png"    # labeled side-by-side TX / RX image
SHOW_RESULT = True              # pop up COMPARE_OUT when the run finishes

# tones: keep all four inside the pair's usable band (check with the SNR sweep)
PAIR_A      = (55.5e3, 56.5e3)  # (bit 0, bit 1) for even symbols
PAIR_B      = (57.5e3, 58.5e3)  # (bit 0, bit 1) for odd symbols
TS          = 2e-3              # symbol time, s (500 bps). Raise if BER is bad.
SKIP_FRAC   = 0.15              # fraction of each symbol ignored (transient/ISI)

CHIRP_F     = (53e3, 61e3)
CHIRP_T     = 5e-3
GAP_T       = 20e-3             # silence after chirp so its reverb decays
TAIL_T      = 20e-3

BYTES_PER_FRAME = 64
MAX_TRIES   = 3

ARB_SRATE   = 400e3             # AO update rate, Sa/s
DRIVE_VPP   = 0.5               # Vpp at AO0 (into amp input)
MAX_VPP     = 0.5               # hard limit on AO0, enforced in software and by DAQmx

AI_SRATE    = 500e3             # AI0 sample rate, Sa/s (6341 max for one channel)
AI_RANGE    = 0.2               # AI0 input range, +/- V (smallest 6341 range)
RX_EXTRA_T  = 50e-3             # record this long past the end of the frame

RX_BAND     = (53e3, 61e3)

CSV_OUT     = False             # True: just write each frame to a CSV, no instruments
CSV_DIR     = "D:/csv_frames"   # one file per frame, samples normalized to +/-1
# -----------------------------------------------

TRAIN_BITS = [0, 0, 1, 1, 0, 0, 1, 1]   # hits each of the 4 tones twice
HDR_BITS = 32
CRC_BITS = 16


def bytes_to_bits(b):
    return list(np.unpackbits(np.frombuffer(b, dtype=np.uint8)))


def bits_to_bytes(bits):
    return np.packbits(np.array(bits, dtype=np.uint8)).tobytes()


def tone(i, bit):
    pair = PAIR_A if i % 2 == 0 else PAIR_B
    return pair[bit]


# ---------------- TX ----------------
def build_frame_bits(idx, payload):
    hdr = idx.to_bytes(2, "big") + len(payload).to_bytes(2, "big")
    crc = binascii.crc_hqx(hdr + payload, 0xFFFF).to_bytes(2, "big")
    return TRAIN_BITS + bytes_to_bits(hdr + payload + crc)


def chirp(fs):
    t = np.arange(int(CHIRP_T * fs)) / fs
    k = (CHIRP_F[1] - CHIRP_F[0]) / CHIRP_T
    x = np.sin(2 * np.pi * (CHIRP_F[0] * t + 0.5 * k * t ** 2))
    return x * np.hanning(len(x)) ** 0.25          # soft edges


def build_waveform(bits, fs):
    ns = int(round(TS * fs))
    f_inst = np.concatenate([np.full(ns, tone(i, b)) for i, b in enumerate(bits)])
    phase = 2 * np.pi * np.cumsum(f_inst) / fs      # continuous phase
    data = np.sin(phase)
    ramp = min(ns // 4, 50)
    win = np.ones(len(data))
    win[:ramp] = np.linspace(0, 1, ramp)
    win[-ramp:] = np.linspace(1, 0, ramp)
    data *= win
    return np.concatenate([chirp(fs),
                           np.zeros(int(GAP_T * fs)),
                           data,
                           np.zeros(int(TAIL_T * fs))])


def tx_buffer(wave):
    if DRIVE_VPP > MAX_VPP:
        raise ValueError(f"DRIVE_VPP {DRIVE_VPP} V exceeds MAX_VPP {MAX_VPP} V")
    data = np.clip(wave / np.max(np.abs(wave)) * DRIVE_VPP / 2,
                   -MAX_VPP / 2, MAX_VPP / 2)
    data[-1] = 0                                    # leave AO0 at 0 V
    if np.ptp(data) > MAX_VPP:
        raise ValueError(f"AO0 buffer is {np.ptp(data):.4f} Vpp, over {MAX_VPP} V")
    return data


# ---------------- TX + RX ----------------
def txrx(tx):
    """Play tx on AO0 and record AI0, both started by the same AO start trigger."""
    n_ai = int((len(tx) / ARB_SRATE + RX_EXTRA_T) * AI_SRATE)
    with nidaqmx.Task() as ao, nidaqmx.Task() as ai:
        # DAQmx rejects the write if any sample is outside these limits
        ao.ao_channels.add_ao_voltage_chan(AO_CHAN, min_val=-MAX_VPP / 2, max_val=MAX_VPP / 2)
        ao.timing.cfg_samp_clk_timing(ARB_SRATE, sample_mode=AcquisitionType.FINITE,
                                      samps_per_chan=len(tx))
        ao.write(tx, auto_start=False)

        ai.ai_channels.add_ai_voltage_chan(AI_CHAN, terminal_config=AI_TERM,
                                           min_val=-AI_RANGE, max_val=AI_RANGE)
        ai.timing.cfg_samp_clk_timing(AI_SRATE, sample_mode=AcquisitionType.FINITE,
                                      samps_per_chan=n_ai)
        ai.triggers.start_trigger.cfg_dig_edge_start_trig(f"/{DEV}/ao/StartTrigger")

        ai.start()                                  # waits for the AO start trigger
        ao.start()
        v = np.array(ai.read(number_of_samples_per_channel=n_ai,
                             timeout=n_ai / AI_SRATE + 5))
        ao.wait_until_done(timeout=5)
    t = np.arange(n_ai) / AI_SRATE                  # t = 0 at the first TX sample
    return t, v, AI_SRATE


# ---------------- RX ----------------
def demod(t, v, fs, n_bits):
    sos = butter(4, RX_BAND, btype="band", fs=fs, output="sos")
    y = sosfiltfilt(sos, v)

    tpl = chirp(fs)
    corr = np.abs(fftconvolve(y, tpl[::-1], mode="valid"))
    n0 = int(np.argmax(corr))

    ns = TS * fs
    start = n0 + (CHIRP_T + GAP_T) * fs
    skip = int(SKIP_FRAC * ns)
    tt = np.arange(int(ns) - skip) / fs

    energies = []
    for i in range(n_bits):
        a = int(round(start + i * ns)) + skip
        seg = y[a:a + len(tt)]
        if len(seg) < len(tt):
            return None, None
        e = [np.abs(np.dot(seg, np.exp(-2j * np.pi * tone(i, b) * tt))) ** 2
             for b in (0, 1)]
        energies.append(e)
    energies = np.array(energies)

    # per-tone gain from training symbols
    gain = {}
    for i, b in enumerate(TRAIN_BITS):
        gain.setdefault(tone(i, b), []).append(energies[i, b])
    gain = {f: np.mean(g) for f, g in gain.items()}

    bits = []
    for i in range(n_bits):
        e0 = energies[i, 0] / gain[tone(i, 0)]
        e1 = energies[i, 1] / gain[tone(i, 1)]
        bits.append(int(e1 > e0))
    return bits, corr[n0]


def parse(bits):
    raw = bits_to_bytes(bits[len(TRAIN_BITS):])
    hdr, crc = raw[:4], raw[-2:]
    idx = int.from_bytes(hdr[:2], "big")
    nbytes = int.from_bytes(hdr[2:4], "big")
    payload = raw[4:4 + nbytes]
    ok = binascii.crc_hqx(hdr + payload, 0xFFFF).to_bytes(2, "big") == crc
    return ok, idx, payload


def write_csvs(frames):
    os.makedirs(CSV_DIR, exist_ok=True)
    for idx, payload in enumerate(frames):
        wave = build_waveform(build_frame_bits(idx, payload), ARB_SRATE)
        wave = wave / np.max(np.abs(wave))
        path = os.path.join(CSV_DIR, f"frame_{idx:03d}.csv")
        np.savetxt(path, wave, fmt="%.5f")
        print(f"  {path}: {len(wave)} samples, {len(wave) / ARB_SRATE:.3f} s")
    print(f"Wrote {len(frames)} files to {CSV_DIR}/ "
          f"(play at {ARB_SRATE:.0f} Sa/s, {DRIVE_VPP} Vpp)")


def compare_image(tx_img, rx_img, gap=24, label_h=44):
    """Side-by-side TX / RX image, scaled up to ~400 px, with a label above each."""
    scale = max(1, round(400 / max(IMG_W, IMG_H)))
    w, h = IMG_W * scale, IMG_H * scale
    out = Image.new("L", (2 * w + gap, h + label_h), 255)
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=28)
    for i, (im, label) in enumerate([(tx_img, "Sent (TX)"), (rx_img, "Received (RX)")]):
        x = i * (w + gap)
        out.paste(im.resize((w, h), Image.NEAREST), (x, label_h))
        draw.text((x + w // 2, label_h // 2), label, fill=0, font=font, anchor="mm")
    return out


# ---------------- main ----------------
def main():
    if CSV_OUT:
        img = Image.open(IMAGE_IN).convert("L").resize((IMG_W, IMG_H))
        data = img.tobytes()
        write_csvs([data[i:i + BYTES_PER_FRAME] for i in range(0, len(data), BYTES_PER_FRAME)])
        return

    for out in (IMAGE_OUT, COMPARE_OUT):
        if os.path.abspath(out) == os.path.abspath(IMAGE_IN):
            raise ValueError(f"{out} would overwrite the input image {IMAGE_IN}")

    img = Image.open(IMAGE_IN).convert("L").resize((IMG_W, IMG_H))
    data = img.tobytes()
    frames = [data[i:i + BYTES_PER_FRAME] for i in range(0, len(data), BYTES_PER_FRAME)]
    rx = bytearray(b"\x80" * len(data))            # gray fill for lost frames

    frame_dur = (CHIRP_T + GAP_T + TAIL_T +
                 (len(TRAIN_BITS) + HDR_BITS + 8 * BYTES_PER_FRAME + CRC_BITS) * TS)
    print(f"{len(frames)} frames, ~{frame_dur:.2f} s air time each, "
          f"{1/TS:.0f} bps raw")

    total_err = total_bits = 0
    for idx, payload in enumerate(frames):
        tx_bits = build_frame_bits(idx, payload)
        tx_data = tx_buffer(build_waveform(tx_bits, ARB_SRATE))

        for attempt in range(1, MAX_TRIES + 1):
            t, v, fs = txrx(tx_data)
            rx_bits, pk = demod(t, v, fs, len(tx_bits))
            if rx_bits is None:
                print(f"frame {idx:3d} try {attempt}: capture too short")
                continue
            errs = int(np.sum(np.array(rx_bits) != np.array(tx_bits)))
            ok, ridx, rpay = parse(rx_bits)
            print(f"frame {idx:3d} try {attempt}: bit errors {errs:4d}/{len(tx_bits)} "
                  f"CRC {'OK ' if ok else 'BAD'}")
            total_err += errs
            total_bits += len(tx_bits)
            if ok and ridx == idx:
                start = idx * BYTES_PER_FRAME
                rx[start:start + len(rpay)] = rpay
                break
        time.sleep(0.1)

    rx_img = Image.frombytes("L", (IMG_W, IMG_H), bytes(rx))
    rx_img.save(IMAGE_OUT)
    side = compare_image(img, rx_img)
    side.save(COMPARE_OUT)
    print(f"\nRaw BER: {total_err}/{total_bits} = {total_err/max(total_bits,1):.2e}")
    print(f"Saved {IMAGE_OUT} and {COMPARE_OUT}")
    if SHOW_RESULT:
        side.show()                                 # opens in the default image viewer


if __name__ == "__main__":
    main()
