#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import librosa
import mido
import numpy as np


MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
     2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
     2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)
NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze BPM, beat positions and key for every WAV file "
            "in a directory."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output_jam/work_wav"),
        help="WAV input directory. Default: output_jam/work_wav",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output_jam/step4_analysis"),
        help="Analysis output directory.",
    )
    parser.add_argument(
        "--meter",
        type=int,
        default=4,
        choices=(3, 4, 6),
        help="Beats per bar. Default: 4",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also scan subdirectories.",
    )
    parser.add_argument(
        "--fixed-tempo",
        action="store_true",
        help="Write one global BPM instead of a beat-by-beat tempo map.",
    )
    return parser.parse_args()


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = (a - np.mean(a)) / (np.std(a) + 1e-12)
    b = (b - np.mean(b)) / (np.std(b) + 1e-12)
    return float(np.dot(a, b) / len(a))


def estimate_key(
    y_harmonic: np.ndarray,
    sr: int,
) -> tuple[str, float]:
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    chroma_vector = np.median(chroma, axis=1)

    peak = float(np.max(chroma_vector))
    if peak > 0:
        chroma_vector /= peak

    candidates: list[tuple[str, float]] = []

    for tonic in range(12):
        candidates.append(
            (
                f"{NOTE_NAMES[tonic]} major",
                normalized_correlation(
                    chroma_vector,
                    np.roll(MAJOR_PROFILE, tonic),
                ),
            )
        )
        candidates.append(
            (
                f"{NOTE_NAMES[tonic]} minor",
                normalized_correlation(
                    chroma_vector,
                    np.roll(MINOR_PROFILE, tonic),
                ),
            )
        )

    candidates.sort(key=lambda item: item[1], reverse=True)
    best_key, best_score = candidates[0]
    second_score = candidates[1][1]

    confidence = np.clip(
        (best_score - second_score + 0.1) / 0.3,
        0.0,
        1.0,
    )
    return best_key, float(confidence)


def estimate_downbeat_offset(
    beat_frames: np.ndarray,
    onset_envelope: np.ndarray,
    meter: int,
) -> int:
    if len(beat_frames) < meter:
        return 0

    phase_scores = np.zeros(meter, dtype=np.float64)

    for beat_index, frame in enumerate(beat_frames):
        start = max(0, int(frame) - 1)
        end = min(len(onset_envelope), int(frame) + 2)
        strength = float(np.max(onset_envelope[start:end]))
        phase_scores[beat_index % meter] += strength

    return int(np.argmax(phase_scores))


def calculate_tempo_stats(
    beat_times: np.ndarray,
    fallback_bpm: float,
) -> tuple[np.ndarray, float, float, bool]:
    if len(beat_times) < 2:
        return (
            np.array([], dtype=np.float64),
            fallback_bpm,
            0.0,
            True,
        )

    intervals = np.diff(beat_times)
    intervals = intervals[intervals > 0]

    if len(intervals) == 0:
        return (
            np.array([], dtype=np.float64),
            fallback_bpm,
            0.0,
            True,
        )

    local_bpms = 60.0 / intervals
    median_bpm = float(np.median(local_bpms))

    filtered = local_bpms[
        (local_bpms >= median_bpm * 0.65)
        & (local_bpms <= median_bpm * 1.45)
    ]
    if len(filtered) == 0:
        filtered = local_bpms

    global_bpm = float(np.median(filtered))
    variation_percent = float(
        100.0 * np.std(filtered) / max(global_bpm, 1e-9)
    )

    return (
        local_bpms,
        global_bpm,
        variation_percent,
        variation_percent < 2.0,
    )


def export_beats_csv(
    output_path: Path,
    beat_times: np.ndarray,
    local_bpms: np.ndarray,
    downbeat_offset: int,
    meter: int,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "detected_beat",
                "time_seconds",
                "bar",
                "beat_in_bar",
                "is_downbeat",
                "local_bpm",
            ]
        )

        for index, beat_time in enumerate(beat_times):
            relative = index - downbeat_offset

            if relative >= 0:
                bar = relative // meter + 1
            else:
                bar = 0

            beat_in_bar = relative % meter + 1
            local_bpm = (
                float(local_bpms[index])
                if index < len(local_bpms)
                else (
                    float(local_bpms[-1])
                    if len(local_bpms)
                    else None
                )
            )

            writer.writerow(
                [
                    index + 1,
                    f"{float(beat_time):.6f}",
                    bar,
                    beat_in_bar,
                    int(beat_in_bar == 1),
                    (
                        f"{local_bpm:.6f}"
                        if local_bpm is not None
                        else ""
                    ),
                ]
            )


def seconds_to_ticks(
    seconds: float,
    bpm: float,
    ticks_per_beat: int,
) -> int:
    return max(
        0,
        int(round(seconds * bpm / 60.0 * ticks_per_beat)),
    )


def export_tempo_midi(
    output_path: Path,
    beat_times: np.ndarray,
    local_bpms: np.ndarray,
    global_bpm: float,
    meter: int,
    downbeat_offset: int,
    fixed_tempo: bool,
) -> None:
    ticks_per_beat = 480
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)

    track.append(
        mido.MetaMessage(
            "track_name",
            name="Tempo Map",
            time=0,
        )
    )
    track.append(
        mido.MetaMessage(
            "time_signature",
            numerator=meter,
            denominator=4,
            clocks_per_click=24,
            notated_32nd_notes_per_beat=8,
            time=0,
        )
    )
    track.append(
        mido.MetaMessage(
            "set_tempo",
            tempo=mido.bpm2tempo(global_bpm),
            time=0,
        )
    )

    if len(beat_times) == 0:
        track.append(mido.MetaMessage("end_of_track", time=0))
        midi.save(output_path)
        return

    first_tick = seconds_to_ticks(
        float(beat_times[0]),
        global_bpm,
        ticks_per_beat,
    )
    track.append(
        mido.MetaMessage(
            "marker",
            text="First detected beat",
            time=first_tick,
        )
    )

    if fixed_tempo:
        if downbeat_offset < len(beat_times):
            offset_seconds = float(
                beat_times[downbeat_offset] - beat_times[0]
            )
            downbeat_tick = seconds_to_ticks(
                offset_seconds,
                global_bpm,
                ticks_per_beat,
            )
            track.append(
                mido.MetaMessage(
                    "marker",
                    text="Estimated bar 1",
                    time=downbeat_tick,
                )
            )

        track.append(mido.MetaMessage("end_of_track", time=0))
        midi.save(output_path)
        return

    for index in range(len(beat_times) - 1):
        bpm = float(local_bpms[index])

        if not math.isfinite(bpm) or bpm <= 0:
            bpm = global_bpm

        track.append(
            mido.MetaMessage(
                "set_tempo",
                tempo=mido.bpm2tempo(bpm),
                time=0 if index == 0 else ticks_per_beat,
            )
        )

        if index == downbeat_offset:
            track.append(
                mido.MetaMessage(
                    "marker",
                    text="Estimated bar 1",
                    time=0,
                )
            )

    track.append(
        mido.MetaMessage(
            "end_of_track",
            time=ticks_per_beat,
        )
    )
    midi.save(output_path)


def analyze_wav(
    audio_path: Path,
    output_dir: Path,
    meter: int,
    fixed_tempo: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    if len(y) == 0:
        raise ValueError("Audio file is empty")

    duration = float(librosa.get_duration(y=y, sr=sr))
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    hop_length = 512
    onset_envelope = librosa.onset.onset_strength(
        y=y_percussive,
        sr=sr,
        hop_length=hop_length,
        aggregate=np.median,
    )

    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=hop_length,
        units="frames",
        trim=False,
    )

    fallback_bpm = float(np.asarray(tempo).reshape(-1)[0])
    beat_frames = np.asarray(beat_frames, dtype=int)
    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sr,
        hop_length=hop_length,
    )

    local_bpms, global_bpm, variation, likely_fixed = (
        calculate_tempo_stats(beat_times, fallback_bpm)
    )

    downbeat_offset = (
        estimate_downbeat_offset(
            beat_frames,
            onset_envelope,
            meter,
        )
        if len(beat_times)
        else 0
    )

    estimated_key, key_confidence = estimate_key(
        y_harmonic,
        sr,
    )

    first_downbeat = (
        float(beat_times[downbeat_offset])
        if downbeat_offset < len(beat_times)
        else None
    )

    report = {
        "source_audio": str(audio_path),
        "duration_seconds": round(duration, 6),
        "sample_rate_used": sr,
        "meter": f"{meter}/4",
        "detected_beat_count": int(len(beat_times)),
        "global_bpm": round(global_bpm, 4),
        "tempo_variation_percent": round(variation, 4),
        "likely_fixed_tempo": likely_fixed,
        "estimated_key": estimated_key,
        "key_confidence_relative": round(key_confidence, 4),
        "first_detected_beat_seconds": (
            round(float(beat_times[0]), 6)
            if len(beat_times)
            else None
        ),
        "estimated_first_downbeat_seconds": (
            round(first_downbeat, 6)
            if first_downbeat is not None
            else None
        ),
        "tempo_midi_mode": (
            "fixed" if fixed_tempo else "beat-by-beat variable"
        ),
    }

    (output_dir / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    export_beats_csv(
        output_dir / "beats.csv",
        beat_times,
        local_bpms,
        downbeat_offset,
        meter,
    )
    export_tempo_midi(
        output_dir / "tempo_map.mid",
        beat_times,
        local_bpms,
        global_bpm,
        meter,
        downbeat_offset,
        fixed_tempo,
    )

    print(
        f"[OK] BPM={global_bpm:.2f}, "
        f"key={estimated_key}, "
        f"output={output_dir}"
    )
    return report


def find_wav_files(
    input_dir: Path,
    recursive: bool,
) -> list[Path]:
    iterator = (
        input_dir.rglob("*")
        if recursive
        else input_dir.iterdir()
    )

    return sorted(
        (
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() == ".wav"
        ),
        key=lambda path: str(path).lower(),
    )


def output_name_for(
    wav_path: Path,
    input_dir: Path,
) -> str:
    relative = wav_path.relative_to(input_dir).with_suffix("")
    return "__".join(relative.parts)


def resolve_project_path(
    path: Path,
    project_root: Path,
) -> Path:
    if not path.is_absolute():
        path = project_root / path
    return path.expanduser().resolve()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_dir = resolve_project_path(args.input_dir, project_root)
    output_root = resolve_project_path(args.out, project_root)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    wav_files = find_wav_files(input_dir, args.recursive)

    if not wav_files:
        raise SystemExit(f"No WAV files found in: {input_dir}")

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input: {input_dir}")
    print(f"[INFO] Found {len(wav_files)} WAV file(s)")
    print(f"[INFO] Output: {output_root}")

    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for index, wav_path in enumerate(wav_files, start=1):
        print()
        print(f"[{index}/{len(wav_files)}] {wav_path.name}")

        file_output_dir = (
            output_root / output_name_for(wav_path, input_dir)
        )

        try:
            report = analyze_wav(
                wav_path,
                file_output_dir,
                args.meter,
                args.fixed_tempo,
            )
            results.append(report)
        except Exception as error:
            print(
                f"[ERROR] Failed to analyze {wav_path}: {error}",
                file=sys.stderr,
            )
            failures.append(
                {
                    "source_audio": str(wav_path),
                    "error": str(error),
                }
            )

    summary = {
        "input_directory": str(input_dir),
        "output_directory": str(output_root),
        "total": len(wav_files),
        "succeeded": len(results),
        "failed": len(failures),
        "results": results,
        "failures": failures,
    }

    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"[SUMMARY] total={len(wav_files)}, "
        f"succeeded={len(results)}, "
        f"failed={len(failures)}"
    )
    print(f"[OK] Summary: {output_root / 'summary.json'}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
