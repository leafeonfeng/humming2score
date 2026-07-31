from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


AUDIO_EXTENSIONS = {
    ".m4a",
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".opus",
    ".aiff",
    ".aif",
}

EXPECTED_STEM_NAMES = [
    "vocals",
    "drums",
    "bass",
    "other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 1 for band jam recordings: convert audio to WAV and "
            "separate vocals / drums / bass / other with Demucs."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input_jam"),
        help="Directory containing Jam recordings. Default: input_jam",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_jam"),
        help="Output root directory. Default: output_jam",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="htdemucs",
        help=(
            "Demucs model name. "
            "Recommended default: htdemucs"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Demucs device. "
            "auto = let Demucs choose; "
            "cuda = force NVIDIA GPU; "
            "cpu = force CPU. "
            "Default: auto"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-run Demucs even if the expected stem directory already "
            "exists. Default: skip existing Demucs output."
        ),
    )

    return parser.parse_args()


def format_command(command: list[str]) -> str:
    """Format a command for readable terminal output."""
    return " ".join(
        f'"{part}"' if " " in str(part) else str(part)
        for part in command
    )


def check_requirements() -> None:
    """Confirm that FFmpeg, FFprobe and Demucs are available."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg was not found in PATH.\n"
            "Install FFmpeg and reopen the terminal.\n"
            "Windows example:\n"
            "  winget install Gyan.FFmpeg"
        )

    if shutil.which("ffprobe") is None:
        raise RuntimeError(
            "FFprobe was not found in PATH.\n"
            "FFprobe is normally included with FFmpeg.\n"
            "Reinstall FFmpeg or fix PATH."
        )

    result = subprocess.run(
        [sys.executable, "-m", "demucs", "--help"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Demucs is not available in the current Python environment.\n"
            "Install it with:\n"
            "  python -m pip install demucs"
        )


def check_cuda_if_requested(device: str) -> None:
    """Show CUDA state when CUDA was explicitly requested."""
    if device != "cuda":
        return

    try:
        import torch

    except ImportError as error:
        raise RuntimeError(
            "PyTorch is not installed, so CUDA cannot be checked."
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError(
            "You selected --device cuda, but PyTorch cannot access CUDA.\n"
            "Check with:\n"
            "  python -c \"import torch; print(torch.cuda.is_available())\"\n"
            "\n"
            "Possible solutions:\n"
            "  1. Fix NVIDIA driver / PyTorch CUDA compatibility.\n"
            "  2. Run with --device cpu."
        )

    gpu_name = torch.cuda.get_device_name(0)

    print(
        f"[INFO] CUDA available. Using GPU: {gpu_name}",
        flush=True,
    )


def collect_audio_files(input_dir: Path) -> list[Path]:
    """Collect supported audio files directly in input_dir."""
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def get_audio_duration_seconds(source_path: Path) -> float:
    """Read audio duration through FFprobe."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    duration_text = result.stdout.strip()

    try:
        duration_seconds = float(duration_text)

    except ValueError as error:
        raise RuntimeError(
            "Could not read source audio duration: "
            f"{source_path.name}"
        ) from error

    if duration_seconds <= 0:
        raise RuntimeError(
            "Audio duration is invalid: "
            f"{source_path.name}"
        )

    return duration_seconds


def format_seconds(seconds: float) -> str:
    """Format seconds as MM:SS."""
    total_seconds = max(0, int(seconds))

    minutes = total_seconds // 60
    remaining_seconds = total_seconds % 60

    return f"{minutes:02d}:{remaining_seconds:02d}"


def convert_audio_to_wav(
    source_path: Path,
    wav_output_path: Path,
) -> None:
    """Convert input audio to 44.1kHz stereo WAV with percentage progress."""
    wav_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if wav_output_path.exists():
        print(
            "  [SKIP] WAV already exists: "
            f"{wav_output_path.name}",
            flush=True,
        )
        return

    print(
        "  [STEP 1/2] Converting source audio to WAV: "
        f"{source_path.name}",
        flush=True,
    )

    total_duration_seconds = get_audio_duration_seconds(source_path)

    print(
        "  [INFO] Source duration: "
        f"{format_seconds(total_duration_seconds)} "
        f"({total_duration_seconds:.1f} seconds)",
        flush=True,
    )

    # FFmpeg writes machine-readable progress to stdout.
    # Normal verbose progress display is disabled with -nostats.
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(source_path),
        "-ar",
        "44100",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        "-progress",
        "pipe:1",
        "-nostats",
        str(wav_output_path),
    ]

    print("  [CMD]", flush=True)
    print(f"  {format_command(command)}", flush=True)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError(
            "Could not receive FFmpeg progress output."
        )

    latest_processed_seconds = 0.0

    for raw_line in process.stdout:
        line = raw_line.strip()

        if "=" not in line:
            continue

        key, value = line.split("=", maxsplit=1)

        if key == "out_time_us":
            try:
                latest_processed_seconds = int(value) / 1_000_000

            except ValueError:
                continue

        if key == "progress":
            progress_fraction = min(
                1.0,
                latest_processed_seconds / total_duration_seconds,
            )

            percentage = progress_fraction * 100

            print(
                "\r"
                "  [WAV] "
                f"{percentage:6.2f}% | "
                f"{format_seconds(latest_processed_seconds)} / "
                f"{format_seconds(total_duration_seconds)}",
                end="",
                flush=True,
            )

    return_code = process.wait()

    print("", flush=True)

    if return_code != 0:
        raise RuntimeError(
            "FFmpeg conversion failed. "
            f"Exit code: {return_code}"
        )

    if not wav_output_path.exists():
        raise RuntimeError(
            "FFmpeg completed but did not create WAV output:\n"
            f"{wav_output_path}"
        )

    print(
        "  [OK] WAV conversion completed: "
        f"{wav_output_path.name}",
        flush=True,
    )


def run_demucs_with_progress(
    command: list[str],
    heartbeat_label: str,
) -> None:
    """Run Demucs while showing its native logs and elapsed-time heartbeats.

    Demucs may display a tqdm progress bar when supported by the terminal.
    This function also prints an elapsed-time status every 5 seconds, so
    CPU/GPU processing never appears completely stuck.
    """
    print("  [CMD]", flush=True)
    print(f"  {format_command(command)}", flush=True)

    start_time = time.monotonic()
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        next_message_after_seconds = 5

        while not stop_heartbeat.wait(timeout=1.0):
            elapsed_seconds = time.monotonic() - start_time

            if elapsed_seconds >= next_message_after_seconds:
                print(
                    "\n"
                    f"  [RUNNING] {heartbeat_label} | "
                    f"elapsed {format_seconds(elapsed_seconds)} | "
                    "GPU/CPU is still processing...",
                    flush=True,
                )

                next_message_after_seconds += 5

    heartbeat_thread = threading.Thread(
        target=heartbeat,
        daemon=True,
    )

    heartbeat_thread.start()

    try:
        # stdout/stderr are intentionally inherited by the terminal.
        # This lets Demucs -v logs and its own progress indicators display.
        process = subprocess.Popen(
            command,
            stdout=None,
            stderr=None,
        )

        return_code = process.wait()

    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1.0)

    elapsed_seconds = time.monotonic() - start_time

    print(
        f"  [INFO] Demucs process finished in "
        f"{format_seconds(elapsed_seconds)}.",
        flush=True,
    )

    if return_code != 0:
        raise subprocess.CalledProcessError(
            returncode=return_code,
            cmd=command,
        )


def find_demucs_stem_directory(
    demucs_output_root: Path,
    model_name: str,
    wav_path: Path,
) -> Path:
    """Find the stem folder generated by Demucs."""
    expected_path = (
        demucs_output_root
        / model_name
        / wav_path.stem
    )

    if expected_path.exists():
        return expected_path

    model_output_dir = demucs_output_root / model_name

    if not model_output_dir.exists():
        raise FileNotFoundError(
            "Demucs model output directory was not created:\n"
            f"{model_output_dir}"
        )

    possible_dirs = sorted(
        path
        for path in model_output_dir.iterdir()
        if path.is_dir()
    )

    if len(possible_dirs) == 1:
        return possible_dirs[0]

    raise FileNotFoundError(
        "Could not identify the Demucs stem output directory.\n"
        f"Expected: {expected_path}"
    )


def validate_stems(stem_dir: Path) -> dict[str, Path]:
    """Validate expected Demucs 4-stem output."""
    stems: dict[str, Path] = {}

    for stem_name in EXPECTED_STEM_NAMES:
        stem_path = stem_dir / f"{stem_name}.wav"

        if not stem_path.exists():
            raise FileNotFoundError(
                "Missing expected Demucs stem:\n"
                f"{stem_path}"
            )

        stems[stem_name] = stem_path

    return stems


def copy_stems_to_project_layout(
    stems: dict[str, Path],
    source_name: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Copy stems into a stable project directory.

    Final output:

    output_jam/
    └─ by_source/
       └─ <source_name>/
          └─ stems/
             ├─ vocals.wav
             ├─ drums.wav
             ├─ bass.wav
             └─ other.wav
    """
    target_stem_dir = (
        output_dir
        / "by_source"
        / source_name
        / "stems"
    )

    target_stem_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied_stems: dict[str, Path] = {}

    for stem_name, source_stem_path in stems.items():
        target_path = target_stem_dir / f"{stem_name}.wav"

        shutil.copy2(
            source_stem_path,
            target_path,
        )

        copied_stems[stem_name] = target_path

    return copied_stems


def write_stem_readme(
    source_name: str,
    copied_stems: dict[str, Path],
    output_dir: Path,
    model_name: str,
    device: str,
) -> None:
    """Write information about generated stems."""
    source_dir = output_dir / "by_source" / source_name
    readme_path = source_dir / "STEMS_README.txt"

    content = f"""Band Jam Stem Separation Result
================================

Source:
{source_name}

Demucs model:
{model_name}

Requested device:
{device}

Generated stems:
- vocals.wav : Primarily vocals. May contain guitar/keyboard/drum leakage.
- drums.wav  : Primarily drum kit and percussion.
- bass.wav   : Primarily bass guitar and low-frequency bass content.
- other.wav  : Guitar, keyboard, synth, residual instruments and leakage.

Important:
- Stem separation is not equivalent to having original multitrack recordings.
- other.wav is not automatically a clean guitar-only or keyboard-only stem.
- The following transcription steps should process each stem differently.

Files:
"""

    for stem_name in EXPECTED_STEM_NAMES:
        stem_path = copied_stems[stem_name]
        content += f"- {stem_name:7s}: {stem_path.name}\n"

    readme_path.write_text(
        content,
        encoding="utf-8",
    )


def separate_one_jam_file(
    source_path: Path,
    output_dir: Path,
    model_name: str,
    device: str,
    overwrite: bool,
) -> None:
    """Convert one Jam recording and perform Demucs 4-stem separation."""
    source_name = source_path.stem

    work_wav_dir = output_dir / "work_wav"
    demucs_output_root = output_dir / "_demucs_raw"

    wav_path = work_wav_dir / f"{source_name}.wav"

    convert_audio_to_wav(
        source_path=source_path,
        wav_output_path=wav_path,
    )

    print(
        "  [STEP 2/2] Separating vocals / drums / bass / other: "
        f"{source_name}",
        flush=True,
    )

    print(
        "  [INFO] Demucs will use GPU when --device cuda is selected.",
        flush=True,
    )

    expected_stem_dir = (
        demucs_output_root
        / model_name
        / source_name
    )

    if expected_stem_dir.exists() and not overwrite:
        print(
            "  [SKIP] Existing Demucs result found: "
            f"{expected_stem_dir}",
            flush=True,
        )

    else:
        command = [
            sys.executable,
            "-u",
            "-m",
            "demucs",
            "-v",
            "-n",
            model_name,
            "--out",
            str(demucs_output_root),
        ]

        if device != "auto":
            command.extend(
                [
                    "-d",
                    device,
                ]
            )

        command.append(str(wav_path))

        device_label = (
            "CUDA GPU"
            if device == "cuda"
            else "CPU"
            if device == "cpu"
            else "auto-selected device"
        )

        run_demucs_with_progress(
            command=command,
            heartbeat_label=(
                f"Demucs stem separation ({device_label}) "
                f"for {source_path.name}"
            ),
        )

    raw_stem_dir = find_demucs_stem_directory(
        demucs_output_root=demucs_output_root,
        model_name=model_name,
        wav_path=wav_path,
    )

    stems = validate_stems(raw_stem_dir)

    copied_stems = copy_stems_to_project_layout(
        stems=stems,
        source_name=source_name,
        output_dir=output_dir,
    )

    write_stem_readme(
        source_name=source_name,
        copied_stems=copied_stems,
        output_dir=output_dir,
        model_name=model_name,
        device=device,
    )

    print("  [OK] Generated stem files:", flush=True)

    for stem_name in EXPECTED_STEM_NAMES:
        print(
            f"       {stem_name:7s} -> "
            f"{copied_stems[stem_name]}",
            flush=True,
        )


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        print(
            "[ERROR] Input directory does not exist:\n"
            f"  {args.input_dir.resolve()}"
        )
        sys.exit(1)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        check_requirements()
        check_cuda_if_requested(args.device)

    except RuntimeError as error:
        print("[ERROR]")
        print(error)
        sys.exit(1)

    audio_files = collect_audio_files(args.input_dir)

    if not audio_files:
        print(
            "[WARN] No supported audio files found in:\n"
            f"  {args.input_dir.resolve()}"
        )

        print(
            "[INFO] Supported extensions: "
            + ", ".join(sorted(AUDIO_EXTENSIONS))
        )

        return

    print("=" * 76)
    print("[INFO] Step 1: Jam Recording -> WAV -> Demucs 4-Stem Separation")
    print(f"[INFO] Input : {args.input_dir.resolve()}")
    print(f"[INFO] Output: {args.output_dir.resolve()}")
    print(f"[INFO] Model : {args.model}")
    print(f"[INFO] Device: {args.device}")
    print("[INFO] Stems : vocals / drums / bass / other")
    print("[INFO] Progress:")
    print("       WAV conversion = percentage progress")
    print("       Demucs         = verbose output + 5-second heartbeat")
    print("=" * 76)

    success_count = 0

    for index, source_path in enumerate(audio_files, start=1):
        print()
        print(
            f"[{index}/{len(audio_files)}] Processing: "
            f"{source_path.name}",
            flush=True,
        )

        try:
            separate_one_jam_file(
                source_path=source_path,
                output_dir=args.output_dir,
                model_name=args.model,
                device=args.device,
                overwrite=args.overwrite,
            )

            success_count += 1

        except subprocess.CalledProcessError as error:
            print(
                "  [FAIL] Demucs command failed with exit code: "
                f"{error.returncode}",
                flush=True,
            )

        except Exception as error:
            print(
                f"  [FAIL] {error}",
                flush=True,
            )

    print()
    print("=" * 76)
    print(
        "[DONE] Successfully processed: "
        f"{success_count}/{len(audio_files)}",
        flush=True,
    )

    print()
    print("[NEXT] Check generated stems here:", flush=True)
    print(
        "       output_jam/by_source/<jam_name>/stems/",
        flush=True,
    )

    print()
    print("       Expected files:", flush=True)

    for stem_name in EXPECTED_STEM_NAMES:
        print(
            f"       - {stem_name}.wav",
            flush=True,
        )

    print("=" * 76)


if __name__ == "__main__":
    main()
