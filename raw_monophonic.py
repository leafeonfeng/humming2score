from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Do not open GUI windows while rendering PNG files.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import pretty_midi
from basic_pitch.inference import predict


AUDIO_EXTENSIONS = {".mp3"}

OUTPUT_FOLDERS = {
    "basic_pitch_original_midi": "01_basic_pitch_original_midi",
    "raw_monophonic_midi": "02_raw_monophonic_midi",
    "raw_notes_csv": "03_raw_notes_csv",
    "raw_pianoroll_png": "04_raw_pianoroll_png",
    "raw_pianoroll_event_grid_csv": "05_raw_pianoroll_event_grid_csv",
}

RAW_CSV_FIELD_ORDER = [
    "note_index",
    "note_name",
    "midi_pitch",
    "velocity",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
]


@dataclass
class RawNote:
    """A note whose timing remains in original audio seconds."""

    start_seconds: float
    end_seconds: float
    duration_seconds: float
    pitch: int
    note_name: str
    velocity: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract raw monophonic notes from MP3 files. "
            "No BPM, no beat conversion, and no rhythm quantization."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="MP3 input directory. Default: input",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_raw"),
        help="Output root directory. Default: output_raw",
    )

    parser.add_argument(
        "--min-duration-ms",
        type=float,
        default=45.0,
        help=(
            "Discard Basic Pitch notes shorter than this duration. "
            "Default: 45 milliseconds"
        ),
    )

    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.5,
        help="Basic Pitch onset threshold. Default: 0.5",
    )

    parser.add_argument(
        "--frame-threshold",
        type=float,
        default=0.3,
        help="Basic Pitch frame threshold. Default: 0.3",
    )

    parser.add_argument(
        "--keep-overlaps",
        action="store_true",
        help=(
            "Keep Basic Pitch overlaps and chords. "
            "Default behavior is strictly monophonic."
        ),
    )

    parser.add_argument(
        "--pitch-min",
        type=int,
        default=0,
        help="Minimum MIDI pitch to retain. Default: 0",
    )

    parser.add_argument(
        "--pitch-max",
        type=int,
        default=127,
        help="Maximum MIDI pitch to retain. Default: 127",
    )

    args = parser.parse_args()

    if args.min_duration_ms < 0:
        parser.error("--min-duration-ms cannot be negative.")

    if not 0 <= args.onset_threshold <= 1:
        parser.error("--onset-threshold must be between 0 and 1.")

    if not 0 <= args.frame_threshold <= 1:
        parser.error("--frame-threshold must be between 0 and 1.")

    if not 0 <= args.pitch_min <= 127:
        parser.error("--pitch-min must be between 0 and 127.")

    if not 0 <= args.pitch_max <= 127:
        parser.error("--pitch-max must be between 0 and 127.")

    if args.pitch_min > args.pitch_max:
        parser.error("--pitch-min cannot be greater than --pitch-max.")

    return args


def collect_mp3_files(input_dir: Path) -> list[Path]:
    """Collect MP3 files directly inside input_dir."""
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def get_basic_pitch_midi(
    source_path: Path,
    onset_threshold: float,
    frame_threshold: float,
    minimum_note_length_ms: float,
) -> pretty_midi.PrettyMIDI:
    """Run Basic Pitch and return its original MIDI result."""
    print(f"  [STEP 1/5] Basic Pitch transcription: {source_path.name}")

    _, midi_data, _ = predict(
        str(source_path),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length_ms,
    )

    return midi_data


def collect_notes_from_midi(
    midi_data: pretty_midi.PrettyMIDI,
    pitch_min: int,
    pitch_max: int,
    min_duration_seconds: float,
) -> list[pretty_midi.Note]:
    """Collect notes from Basic Pitch MIDI and filter invalid notes."""
    notes: list[pretty_midi.Note] = []

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            duration = note.end - note.start

            if duration < min_duration_seconds:
                continue

            if not pitch_min <= note.pitch <= pitch_max:
                continue

            notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end,
                )
            )

    return sorted(
        notes,
        key=lambda item: (
            item.start,
            item.pitch,
            item.end,
        ),
    )


def choose_best_note(
    notes: list[pretty_midi.Note],
) -> pretty_midi.Note:
    """Choose one note from a simultaneous onset group.

    Priority:
    1. Longer duration
    2. Higher velocity
    3. Higher pitch
    """
    return max(
        notes,
        key=lambda item: (
            item.end - item.start,
            item.velocity,
            item.pitch,
        ),
    )


def make_raw_monophonic(
    notes: list[pretty_midi.Note],
) -> list[pretty_midi.Note]:
    """Ensure that there is at most one active note at any time.

    This function does NOT quantize timing.

    It works fully in original seconds:
    - Notes beginning at nearly the same time are treated as one onset group.
    - Only the best note is retained at that onset.
    - When a later note begins, the earlier note is trimmed to end there.
    - The output remains aligned to original Basic Pitch timing.
    """
    if not notes:
        return []

    same_start_tolerance_seconds = 0.015

    notes = sorted(
        notes,
        key=lambda item: (
            item.start,
            item.pitch,
            item.end,
        ),
    )

    # Step 1:
    # Merge nearly simultaneous detected notes into one note.
    selected_onsets: list[pretty_midi.Note] = []

    index = 0

    while index < len(notes):
        anchor_note = notes[index]
        same_start_notes = [anchor_note]
        index += 1

        while (
            index < len(notes)
            and abs(notes[index].start - anchor_note.start)
            <= same_start_tolerance_seconds
        ):
            same_start_notes.append(notes[index])
            index += 1

        selected_note = choose_best_note(same_start_notes)

        selected_onsets.append(
            pretty_midi.Note(
                velocity=selected_note.velocity,
                pitch=selected_note.pitch,
                start=selected_note.start,
                end=selected_note.end,
            )
        )

    # Step 2:
    # Trim the earlier note when the next note starts.
    result: list[pretty_midi.Note] = []

    for note_index, current_note in enumerate(selected_onsets):
        is_last_note = note_index == len(selected_onsets) - 1

        if is_last_note:
            result.append(current_note)
            continue

        next_note = selected_onsets[note_index + 1]

        # No time overlap: preserve exact original duration.
        if current_note.end <= next_note.start:
            result.append(current_note)
            continue

        trimmed_end = next_note.start

        # If trimming would make the note zero or negative duration,
        # discard that note.
        if trimmed_end <= current_note.start:
            continue

        result.append(
            pretty_midi.Note(
                velocity=current_note.velocity,
                pitch=current_note.pitch,
                start=current_note.start,
                end=trimmed_end,
            )
        )

    return result


def convert_to_raw_notes(
    notes: list[pretty_midi.Note],
) -> list[RawNote]:
    """Convert PrettyMIDI notes into serializable RawNote objects."""
    raw_notes: list[RawNote] = []

    for note in notes:
        duration = note.end - note.start

        if duration <= 0:
            continue

        raw_notes.append(
            RawNote(
                start_seconds=note.start,
                end_seconds=note.end,
                duration_seconds=duration,
                pitch=note.pitch,
                note_name=pretty_midi.note_number_to_name(note.pitch),
                velocity=note.velocity,
            )
        )

    return sorted(
        raw_notes,
        key=lambda item: (
            item.start_seconds,
            item.pitch,
            item.end_seconds,
        ),
    )


def write_raw_monophonic_midi(
    output_path: Path,
    notes: list[pretty_midi.Note],
) -> None:
    """Write raw notes to a MIDI file without rhythmic quantization."""
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=120.0)

    instrument = pretty_midi.Instrument(
        program=0,
        is_drum=False,
        name="Raw Monophonic Melody",
    )

    for note in notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start,
                end=note.end,
            )
        )

    midi_data.instruments.append(instrument)
    midi_data.write(str(output_path))


def write_raw_notes_csv(
    output_path: Path,
    raw_notes: list[RawNote],
) -> None:
    """Write one note per column, retaining original seconds."""
    note_headers = [
        f"note_{index:03d}"
        for index in range(1, len(raw_notes) + 1)
    ]

    rows: list[dict[str, str | int]] = []

    for index, note in enumerate(raw_notes, start=1):
        rows.append(
            {
                "note_index": index,
                "note_name": note.note_name,
                "midi_pitch": note.pitch,
                "velocity": note.velocity,
                "start_seconds": f"{note.start_seconds:.6f}",
                "end_seconds": f"{note.end_seconds:.6f}",
                "duration_seconds": f"{note.duration_seconds:.6f}",
            }
        )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(["field", *note_headers])

        for field_name in RAW_CSV_FIELD_ORDER:
            writer.writerow(
                [
                    field_name,
                    *[
                        row.get(field_name, "")
                        for row in rows
                    ],
                ]
            )


def build_event_times(
    raw_notes: list[RawNote],
) -> list[float]:
    """Create exact time columns from every note start and end event."""
    event_times = {
        round(note.start_seconds, 6)
        for note in raw_notes
    }

    event_times.update(
        round(note.end_seconds, 6)
        for note in raw_notes
    )

    return sorted(event_times)


def write_raw_pianoroll_event_grid_csv(
    output_path: Path,
    raw_notes: list[RawNote],
) -> None:
    """Write a non-quantized piano-roll CSV.

    Horizontal axis:
        Exact note event times from the original transcription.

    Vertical axis:
        Pitch.

    Cell values:
        Velocity if that pitch is active at the corresponding event time.

    Unlike the BPM version, this file has no beat grid and no fixed time
    interval. It is based only on actual note starts and note ends.
    """
    if not raw_notes:
        raise ValueError("No notes available for raw event-grid CSV.")

    min_pitch = max(0, min(note.pitch for note in raw_notes) - 2)
    max_pitch = min(127, max(note.pitch for note in raw_notes) + 2)

    event_times = build_event_times(raw_notes)

    headers = [
        f"{event_time:.6f}s"
        for event_time in event_times
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(["pitch_or_time", *headers])

        for pitch in range(max_pitch, min_pitch - 1, -1):
            cells: list[str] = []

            for event_time in event_times:
                active_notes = [
                    note
                    for note in raw_notes
                    if note.pitch == pitch
                    and note.start_seconds <= event_time
                    and event_time < note.end_seconds
                ]

                if active_notes:
                    cells.append(
                        str(
                            max(
                                note.velocity
                                for note in active_notes
                            )
                        )
                    )
                else:
                    cells.append("")

            pitch_name = pretty_midi.note_number_to_name(pitch)

            writer.writerow(
                [
                    f"{pitch_name} (MIDI {pitch})",
                    *cells,
                ]
            )


def render_raw_piano_roll(
    output_path: Path,
    raw_notes: list[RawNote],
) -> None:
    """Render a piano roll using original seconds, with ASCII-only labels."""
    if not raw_notes:
        raise ValueError("No notes available for raw piano-roll PNG.")

    min_pitch = min(note.pitch for note in raw_notes)
    max_pitch = max(note.pitch for note in raw_notes)

    pitch_low = max(0, min_pitch - 2)
    pitch_high = min(127, max_pitch + 2)

    max_end_seconds = max(
        note.end_seconds
        for note in raw_notes
    )

    figure_width = max(
        10,
        min(28, 8 + max_end_seconds * 0.45),
    )

    figure_height = max(
        4,
        min(13, 3 + (pitch_high - pitch_low) * 0.34),
    )

    fig, ax = plt.subplots(
        figsize=(figure_width, figure_height),
    )

    patches = []
    velocities = []

    for note in raw_notes:
        patches.append(
            Rectangle(
                (
                    note.start_seconds,
                    note.pitch - 0.36,
                ),
                note.duration_seconds,
                0.72,
            )
        )
        velocities.append(note.velocity)

    collection = PatchCollection(
        patches,
        cmap="viridis",
        edgecolor="#1d3557",
        linewidth=0.7,
        alpha=0.92,
    )

    collection.set_array(velocities)
    collection.set_clim(0, 127)

    ax.add_collection(collection)

    colorbar = fig.colorbar(
        collection,
        ax=ax,
        pad=0.015,
    )

    colorbar.set_label(
        "MIDI velocity",
        rotation=270,
        labelpad=16,
    )

    ax.set_xlim(-0.05, max_end_seconds + 0.15)
    ax.set_ylim(pitch_low - 0.6, pitch_high + 0.6)

    ax.set_xlabel("Original audio time (seconds)")
    ax.set_ylabel("Pitch")
    ax.set_title("Raw Piano Roll (No BPM / No Quantization)")

    # Light horizontal pitch lines.
    for pitch in range(pitch_low, pitch_high + 1):
        ax.axhline(
            pitch,
            color="#dedede",
            linewidth=0.45,
            zorder=0,
        )

    # Second-level vertical lines only; this is display-only,
    # not a rhythmic grid.
    second_count = int(math.ceil(max_end_seconds))

    for second in range(second_count + 1):
        ax.axvline(
            second,
            color="#d3dce5",
            linewidth=0.7,
            zorder=0,
        )

    pitch_ticks = list(range(pitch_low, pitch_high + 1))

    ax.set_yticks(pitch_ticks)
    ax.set_yticklabels(
        [
            pretty_midi.note_number_to_name(pitch)
            for pitch in pitch_ticks
        ]
    )

    ax.grid(False)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def build_output_paths(
    output_root: Path,
    source_path: Path,
) -> dict[str, dict[str, Path]]:
    """Create per-source and aggregate output paths."""
    base_name = source_path.stem

    source_dir = output_root / "by_source" / base_name
    source_dir.mkdir(parents=True, exist_ok=True)

    aggregate_dirs = {
        key: output_root / folder_name
        for key, folder_name in OUTPUT_FOLDERS.items()
    }

    for folder in aggregate_dirs.values():
        folder.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "basic_pitch_original_midi": (
            source_dir
            / f"{base_name}_01_basic_pitch_original.mid"
        ),
        "raw_monophonic_midi": (
            source_dir
            / f"{base_name}_02_raw_monophonic.mid"
        ),
        "raw_notes_csv": (
            source_dir
            / f"{base_name}_03_raw_notes.csv"
        ),
        "raw_pianoroll_png": (
            source_dir
            / f"{base_name}_04_raw_pianoroll.png"
        ),
        "raw_pianoroll_event_grid_csv": (
            source_dir
            / f"{base_name}_05_raw_pianoroll_event_grid.csv"
        ),
    }

    aggregate_paths = {
        "basic_pitch_original_midi": (
            aggregate_dirs["basic_pitch_original_midi"]
            / f"{base_name}_01_basic_pitch_original.mid"
        ),
        "raw_monophonic_midi": (
            aggregate_dirs["raw_monophonic_midi"]
            / f"{base_name}_02_raw_monophonic.mid"
        ),
        "raw_notes_csv": (
            aggregate_dirs["raw_notes_csv"]
            / f"{base_name}_03_raw_notes.csv"
        ),
        "raw_pianoroll_png": (
            aggregate_dirs["raw_pianoroll_png"]
            / f"{base_name}_04_raw_pianoroll.png"
        ),
        "raw_pianoroll_event_grid_csv": (
            aggregate_dirs["raw_pianoroll_event_grid_csv"]
            / f"{base_name}_05_raw_pianoroll_event_grid.csv"
        ),
    }

    return {
        "source": source_paths,
        "aggregate": aggregate_paths,
    }


def copy_outputs_to_aggregate(
    source_paths: dict[str, Path],
    aggregate_paths: dict[str, Path],
) -> None:
    """Copy each source-folder output to its aggregate type folder."""
    for key, source_file in source_paths.items():
        shutil.copy2(
            source_file,
            aggregate_paths[key],
        )


def process_one_file(
    source_path: Path,
    output_root: Path,
    min_duration_ms: float,
    keep_overlaps: bool,
    onset_threshold: float,
    frame_threshold: float,
    pitch_min: int,
    pitch_max: int,
) -> int:
    """Process one MP3 without BPM conversion or quantization."""
    basic_pitch_midi = get_basic_pitch_midi(
        source_path=source_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length_ms=min_duration_ms,
    )

    print("  [STEP 2/5] Collecting original note timings")

    raw_basic_pitch_notes = collect_notes_from_midi(
        midi_data=basic_pitch_midi,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        min_duration_seconds=min_duration_ms / 1000.0,
    )

    if not raw_basic_pitch_notes:
        raise ValueError("No usable notes were detected.")

    if keep_overlaps:
        final_notes = raw_basic_pitch_notes
        print("  [INFO] Overlaps kept: polyphonic raw output.")
    else:
        print("  [INFO] Making raw output strictly monophonic.")
        final_notes = make_raw_monophonic(raw_basic_pitch_notes)

    if not final_notes:
        raise ValueError("No notes remain after monophonic cleanup.")

    raw_notes = convert_to_raw_notes(final_notes)

    if not raw_notes:
        raise ValueError("No valid notes remain after conversion.")

    output_paths = build_output_paths(
        output_root=output_root,
        source_path=source_path,
    )

    print("  [STEP 3/5] Writing original and monophonic MIDI")

    # Raw Basic Pitch MIDI reference, before monophonic cleanup.
    basic_pitch_midi.write(
        str(output_paths["source"]["basic_pitch_original_midi"])
    )

    # Single-note raw MIDI, preserving original second timing.
    write_raw_monophonic_midi(
        output_path=output_paths["source"]["raw_monophonic_midi"],
        notes=final_notes,
    )

    print("  [STEP 4/5] Writing raw note CSV and event-grid CSV")

    write_raw_notes_csv(
        output_path=output_paths["source"]["raw_notes_csv"],
        raw_notes=raw_notes,
    )

    write_raw_pianoroll_event_grid_csv(
        output_path=(
            output_paths["source"][
                "raw_pianoroll_event_grid_csv"
            ]
        ),
        raw_notes=raw_notes,
    )

    print("  [STEP 5/5] Rendering raw piano-roll PNG")

    render_raw_piano_roll(
        output_path=output_paths["source"]["raw_pianoroll_png"],
        raw_notes=raw_notes,
    )

    copy_outputs_to_aggregate(
        source_paths=output_paths["source"],
        aggregate_paths=output_paths["aggregate"],
    )

    return len(raw_notes)


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        print(
            "[ERROR] Input directory does not exist: "
            f"{args.input_dir.resolve()}"
        )
        sys.exit(1)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mp3_files = collect_mp3_files(args.input_dir)

    if not mp3_files:
        print(
            "[WARN] No MP3 files found in: "
            f"{args.input_dir.resolve()}"
        )
        return

    print("=" * 68)
    print("[INFO] MP3 -> Raw Monophonic MIDI / CSV / PNG")
    print("[INFO] No BPM conversion and no rhythm quantization.")
    print(f"[INFO] Input : {args.input_dir.resolve()}")
    print(f"[INFO] Output: {args.output_dir.resolve()}")
    print(f"[INFO] Pitch range: MIDI {args.pitch_min}-{args.pitch_max}")

    if args.keep_overlaps:
        print("[INFO] Mode: keep overlaps / polyphonic")
    else:
        print("[INFO] Mode: strictly monophonic")

    print("[INFO] Output folders:")
    print("[INFO]   by_source/")
    print("[INFO]   01_basic_pitch_original_midi/")
    print("[INFO]   02_raw_monophonic_midi/")
    print("[INFO]   03_raw_notes_csv/")
    print("[INFO]   04_raw_pianoroll_png/")
    print("[INFO]   05_raw_pianoroll_event_grid_csv/")
    print("=" * 68)

    success_count = 0

    for index, source_path in enumerate(mp3_files, start=1):
        print()
        print(
            f"[{index}/{len(mp3_files)}] Processing: "
            f"{source_path.name}"
        )

        try:
            note_count = process_one_file(
                source_path=source_path,
                output_root=args.output_dir,
                min_duration_ms=args.min_duration_ms,
                keep_overlaps=args.keep_overlaps,
                onset_threshold=args.onset_threshold,
                frame_threshold=args.frame_threshold,
                pitch_min=args.pitch_min,
                pitch_max=args.pitch_max,
            )

            print(
                f"  [OK] Done. Final raw notes: {note_count}"
            )

            success_count += 1

        except Exception as error:
            print(f"  [FAIL] {error}")

    print()
    print("=" * 68)
    print(
        f"[DONE] Processed successfully: "
        f"{success_count}/{len(mp3_files)}"
    )
    print("=" * 68)


if __name__ == "__main__":
    main()
