"""Self-ASR pipeline: transcribe corpus audio into tier 1 text (SYNTHSHEET route B / channel v2).

Runs faster-whisper over WAV files, optionally after telephone degradation, and
writes one JSON per file with segments, word timings and full provenance (model,
settings, degradation chain). Resumable: existing outputs are skipped.

Degradation (default on, per SYNTHSHEET §5): resample to 8 kHz, 300-3400 Hz
band-pass, 8-bit mu-law round-trip, back to 16 kHz. Codec round-trips (AMR-NB/
GSM) are a recorded upgrade for when ffmpeg is available; MUSAN noise likewise.

Default target is the AppTek `test` config (split-channel, one speaker per
file, role known from metadata) — the channel-v2 hypothesis side, aligned later
against the shipped verbatim transcripts. The `diarization` config (merged
mono) transcribes the same way but yields no speaker turns without a
diarisation stage; that stage is future work.

Usage:
  uv run scripts/transcribe.py --limit 2            # smoke test
  uv run scripts/transcribe.py                      # full AppTek test config
  uv run scripts/transcribe.py --no-degrade         # clean-audio contrast pass
"""

import argparse
import ctypes
import glob
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

REPO_ROOT = Path(__file__).resolve().parent.parent
WHISPER_SR = 16_000
PHONE_SR = 8_000


def _preload_nvidia_libs() -> None:
    """ctranslate2 dlopens cuDNN/cuBLAS at model load; the pip nvidia wheels
    are not on the loader path, so preload them with RTLD_GLOBAL."""
    import nvidia  # noqa: F401  (namespace package from the wheels)

    for pkg_dir in nvidia.__path__:
        for lib in sorted(glob.glob(f"{pkg_dir}/*/lib/*.so*")):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def telephone_degrade(audio: np.ndarray, sr: int) -> np.ndarray:
    """Telephone-band degradation: 8 kHz, 300-3400 Hz, mu-law round-trip."""
    x = resample_poly(audio, PHONE_SR, sr)
    sos = butter(4, [300, 3400], btype="bandpass", fs=PHONE_SR, output="sos")
    x = sosfilt(sos, x)
    peak = np.max(np.abs(x)) or 1.0
    x = x / peak
    mu = 255.0
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1.0) * 127.5) / 127.5 - 1.0  # 8-bit quantisation
    x = np.sign(q) * (np.expm1(np.abs(q) * np.log1p(mu))) / mu
    x = x * peak
    return resample_poly(x, WHISPER_SR, PHONE_SR).astype(np.float32)


DEGRADATION_DESC = (
    "resample 8kHz; butterworth bandpass 300-3400Hz; mu-law 8-bit round-trip; "
    "resample 16kHz. No codec round-trip, no additive noise (ffmpeg/MUSAN not "
    "yet installed)."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="apptek_callcenter")
    parser.add_argument("--config", default="test", choices=["test", "diarization"])
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--compute", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--no-degrade", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="stop after N files (0 = all)")
    args = parser.parse_args()

    import faster_whisper

    if args.device == "cuda":
        _preload_nvidia_libs()

    variant = "clean" if args.no_degrade else "degraded"
    raw_dir = REPO_ROOT / "data" / "raw" / args.corpus / args.config
    out_root = REPO_ROOT / "data" / "interim" / f"{args.corpus}_selfasr" / f"{args.config}-{variant}"

    wavs = sorted(raw_dir.glob("*/audio/*.wav"))
    todo = []
    for wav in wavs:
        locale = wav.parent.parent.name
        out = out_root / locale / (wav.stem + ".json")
        if not out.exists():
            todo.append((wav, out))
    print(f"{len(wavs)} wavs on disk, {len(todo)} to transcribe -> {out_root}")
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        return

    model = faster_whisper.WhisperModel(
        args.model,
        device=args.device,
        device_index=args.device_index,
        compute_type=args.compute,
    )
    system_id = f"faster-whisper {faster_whisper.__version__} {args.model} {args.compute}"

    for wav, out in todo:
        t0 = time.time()
        audio, sr = sf.read(wav, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if args.no_degrade:
            if sr != WHISPER_SR:
                audio = resample_poly(audio, WHISPER_SR, sr).astype(np.float32)
        else:
            audio = telephone_degrade(audio, sr)

        segments, info = model.transcribe(
            audio, language="en", word_timestamps=True, vad_filter=False
        )
        seg_out, word_out = [], []
        for s in segments:
            seg_out.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text})
            for w in s.words or []:
                word_out.append(
                    {"word": w.word, "start": round(w.start, 3),
                     "end": round(w.end, 3), "p": round(w.probability, 4)}
                )
        rec = {
            "source_wav": str(wav.relative_to(REPO_ROOT)),
            "asr_system": system_id,
            "degradation": None if args.no_degrade else DEGRADATION_DESC,
            "duration_s": round(len(audio) / WHISPER_SR, 2),
            "language_p": round(info.language_probability, 4),
            "segments": seg_out,
            "words": word_out,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        tmp.rename(out)
        rt = rec["duration_s"] / max(time.time() - t0, 1e-6)
        print(f"{wav.name}: {rec['duration_s']:.0f}s audio, {rt:.1f}x realtime", flush=True)


if __name__ == "__main__":
    main()
