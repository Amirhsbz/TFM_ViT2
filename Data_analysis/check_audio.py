#!/usr/bin/env python3
"""Check whether a saved audio.wav file contains actual sound or is silent.

Usage
-----
# Check a specific file:
    python Data_analysis/check_audio.py shared/data/bc_data/0612_182148/audio.wav

# Auto-find and check the most recent audio.wav under shared/data/bc_data/:
    python Data_analysis/check_audio.py

# List all available audio input devices (to find the right audio_device index):
    python Data_analysis/check_audio.py --list-devices

Output
------
The script prints:
  - File duration, sample rate, bit depth
  - RMS and peak level in dB
  - Fraction of samples above the noise floor
  - A verdict: SILENT / VERY QUIET / QUIET / SOUND PRESENT
  - Per-second RMS bar chart (up to 60 s) so you can see exactly when sound occurs

Typical values
--------------
  > -20 dB  normal speech / ambient noise
  -40 dB    very faint signal (mic gain too low)
  < -60 dB  silent / recording failure (wrong device index or muted mic)

If the file is silent, run --list-devices to find the correct sounddevice index
and update audio_device in run_env_logi.py Args accordingly.
"""

import argparse
import wave
from pathlib import Path

import numpy as np


def check_audio(wav_path: Path) -> None:
    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()  # bytes per sample
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    duration_s = n_frames / sample_rate
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sample_width, np.int16)
    samples = np.frombuffer(raw, dtype=dtype)
    if channels > 1:
        samples = samples.reshape(-1, channels)

    # Normalise to [-1, 1] for comparison across bit depths
    max_val = float(np.iinfo(dtype).max)
    normed = samples.astype(np.float64) / max_val

    rms = float(np.sqrt(np.mean(normed ** 2)))
    peak = float(np.max(np.abs(normed)))
    rms_db = 20 * np.log10(rms + 1e-12)
    peak_db = 20 * np.log10(peak + 1e-12)

    # Count samples above a noise floor (~-60 dB ≈ 0.001 normalised)
    noise_floor = 0.001
    active_fraction = float(np.mean(np.abs(normed) > noise_floor))

    print(f"File       : {wav_path}")
    print(f"Duration   : {duration_s:.2f} s  ({n_frames} frames)")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels   : {channels}  |  Bit depth: {sample_width * 8}-bit")
    print()
    print(f"RMS level  : {rms_db:+.1f} dB  ({rms:.6f})")
    print(f"Peak level : {peak_db:+.1f} dB  ({peak:.6f})")
    print(f"Active frac: {active_fraction * 100:.1f}%  (samples > {noise_floor:.3f} = -{-20*np.log10(noise_floor):.0f} dB)")
    print()

    if rms_db < -60:
        verdict = "SILENT — RMS below -60 dB; likely all zeros or recording failure."
    elif rms_db < -40:
        verdict = "VERY QUIET — signal present but extremely faint (possible mic gain issue)."
    elif rms_db < -20:
        verdict = "QUIET — low-level signal recorded."
    else:
        verdict = "SOUND PRESENT — normal recording level."

    print(f"Verdict: {verdict}")

    # Show per-second RMS to see if any moment has sound
    print()
    chunk = sample_rate
    total_chunks = len(normed) // chunk
    if total_chunks > 0:
        print("Per-second RMS (dB):")
        for i in range(min(total_chunks, 60)):
            seg = normed[i * chunk : (i + 1) * chunk]
            seg_rms = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)
            bar_len = max(0, int((seg_rms + 80) / 2))
            bar = "#" * bar_len
            print(f"  t={i:3d}s  {seg_rms:+6.1f} dB  |{bar}")
        if total_chunks > 60:
            print(f"  ... ({total_chunks - 60} more seconds not shown)")


def list_devices() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed — cannot list devices.")
        return
    devices = sd.query_devices()
    print(f"{'IDX':>4}  {'NAME':<50}  {'IN':>3}  {'OUT':>3}  SR")
    print("-" * 75)
    for i, d in enumerate(devices):
        marker = " <-- default input" if i == sd.default.device[0] else ""
        print(
            f"{i:>4}  {d['name']:<50}  {d['max_input_channels']:>3}  "
            f"{d['max_output_channels']:>3}  {int(d['default_samplerate'])}{marker}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether a trajectory audio.wav contains sound."
    )
    parser.add_argument(
        "wav",
        nargs="?",
        type=Path,
        default=None,
        help="Path to audio.wav. If omitted, scans ./shared/data/bc_data for the most recent one.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List all available audio input devices and exit.",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.wav is not None:
        wav_path = args.wav.expanduser().resolve()
        if not wav_path.exists():
            raise FileNotFoundError(f"File not found: {wav_path}")
        check_audio(wav_path)
    else:
        # Auto-find the most recent audio.wav under the default data directory
        data_root = Path("shared/data/bc_data")
        wavs = sorted(data_root.rglob("audio.wav"))
        if not wavs:
            print(f"No audio.wav files found under {data_root}")
            return
        latest = wavs[-1]
        print(f"No path given — using most recent: {latest}\n")
        check_audio(latest)


if __name__ == "__main__":
    main()
