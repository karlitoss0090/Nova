#!/usr/bin/env python3
"""
Download Kokoro TTS model files.
Usage:  python scripts/download_voices.py

Files downloaded:
  kokoro/kokoro-v1.0.int8.onnx   ~88 MB  — int8-quantised model (CPU-optimised)
  kokoro/voices-v1.0.bin         ~27 MB  — voice embeddings (EN + FR + more)

Source: https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0
"""
import sys
import urllib.request
from pathlib import Path

KOKORO_DIR = Path("kokoro")
BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

FILES = {
    "kokoro-v1.0.int8.onnx": f"{BASE}/kokoro-v1.0.int8.onnx",   # 88 MB
    "voices-v1.0.bin":       f"{BASE}/voices-v1.0.bin",           # 27 MB
}


def _progress(count, block_size, total_size):
    if total_size <= 0:
        return
    pct = min(count * block_size * 100 // total_size, 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    done_mb  = count * block_size / 1024 / 1024
    total_mb = total_size / 1024 / 1024
    print(f"\r  [{bar}] {pct:3d}%  {done_mb:.1f}/{total_mb:.1f} MB", end="", flush=True)


def download(url: str, dest: Path):
    print(f"  ↓ {dest.name}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"\r  ✓ {dest.name} ({size_mb:.1f} MB)   ")
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {dest.name}: {e}") from e


def main():
    KOKORO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n🎙  Kokoro TTS — downloading model files to {KOKORO_DIR.resolve()}\n")

    all_present = True
    for filename, url in FILES.items():
        dest = KOKORO_DIR / filename
        if dest.exists():
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"  ✓ Already exists: {filename} ({size_mb:.1f} MB)")
        else:
            all_present = False
            try:
                download(url, dest)
            except RuntimeError as e:
                print(f"\n  ✗ {e}", file=sys.stderr)
                print(
                    "\n  Check the latest release at:\n"
                    "  https://github.com/thewh1teagle/kokoro-onnx/releases",
                    file=sys.stderr,
                )
                sys.exit(1)

    if all_present:
        print("\n✅  All model files already present — nothing to download.")
    else:
        print("\n✅  Done!")

    print("\nVoices available:")
    print("  English: af_heart (warm, default), af_bella, am_adam, am_michael, bf_emma, bm_george")
    print("  French:  ff_siwis (default)")
    print("\nSet KOKORO_VOICE_EN / KOKORO_VOICE_FR in .env to switch voices.")


if __name__ == "__main__":
    main()
