from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Render PNG files without opening GUI windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

import pretty_midi
from basic_pitch.inference import predict


AUDIO_EXTENSIONS = {".mp3"}

OUTPUT_FOLDERS = {
    "basic_pitch_original_midi": "01_basic_pitch_original_midi",
    "raw_candidates_midi": "02_raw_top_candidates_midi",
    "raw_notes_csv": "03_raw_notes_csv",
    "raw_pianoroll_png": "04_raw_pianoroll_png",
    "raw_pianoroll_event_grid_csv": "05_raw_pianoroll_event_grid_csv",
}

RAW_CSV_FIELD_ORDER = [
    "note_index",
    "note_name",
    "midi_pitch",
    "velocity",
    "confidence",
    "candidate_rank",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
]


@dataclass
class CandidateNote:
    """Basic Pitch candidate note with relative confidence."""

    start_seconds: float
    end_seconds: float
    pitch: int
    velocity: int
    confidence: float


@dataclass
class RawNote:
    """Final note output.

    candidate_rank:
        1 = highest-confidence pitch in this onset group.
        2 = second-highest-confidence pitch in this onset group.
    """

    start_seconds: float
    end_seconds: float
    duration_seconds: float
    pitch: int
    note_name: str
    velocity: int
    confidence: float
    candidate_rank: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract raw Basic Pitch candidates from MP3 files. "
            "No BPM conversion and no rhythmic quantization."
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
        "--max-candidates-per-onset",
        type=int,
        choices=[1, 2],
        default=2,
        help=(
            "How many confidence-ranked pitches to keep per onset group. "
            "1 = only the highest-confidence pitch. "
            "2 = retain highest and second-highest pitches. "
            "Default: 2"
        ),
    )

    parser.add_argument(
        "--same-start-tolerance-ms",
        type=float,
        default=15.0,
        help=(
            "Notes starting within this many milliseconds are treated as "
            "simultaneous candidates. Default: 15"
        ),
    )

    parser.add_argument(
        "--min-duration-ms",
        type=float,
        default=45.0,
        help=(
            "Discard detected notes shorter than this duration. "
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

    if args.same_start_tolerance_ms < 0:
        parser.error("--same-start-tolerance-ms cannot be negative.")

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
    """Collect MP3 files directly in the input directory."""
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def get_basic_pitch_result(
    source_path: Path,
    onset_threshold: float,
    frame_threshold: float,
    minimum_note_length_ms: float,
) -> tuple[pretty_midi.PrettyMIDI, list]:
    """Run Basic Pitch and return MIDI plus note events.

    Typical Basic Pitch note event structure:

        (
            start_seconds,
            end_seconds,
            midi_pitch,
            amplitude,
            pitch_bend,
        )

    amplitude is used as a relative confidence value.
    It is useful for ranking candidates but is not a perfect probability.
    """
    print(f"  [STEP 1/5] Basic Pitch transcription: {source_path.name}")

    _, midi_data, note_events = predict(
        str(source_path),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length_ms,
    )

    return midi_data, note_events


def collect_all_midi_notes(
    midi_data: pretty_midi.PrettyMIDI,
) -> list[pretty_midi.Note]:
    """Collect all non-drum notes from original Basic Pitch MIDI."""
    notes: list[pretty_midi.Note] = []

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue

        notes.extend(instrument.notes)

    return sorted(
        notes,
        key=lambda item: (
            item.start,
            item.end,
            item.pitch,
        ),
    )


def confidence_to_velocity(confidence: float) -> int:
    """Fallback velocity if no matching Basic Pitch MIDI note is found."""
    confidence = max(0.0, min(1.0, confidence))

    velocity = round(1 + confidence * 126)

    return max(1, min(127, velocity))


def find_matching_velocity(
    midi_notes: list[pretty_midi.Note],
    start_seconds: float,
    end_seconds: float,
    pitch: int,
) -> int:
    """Find closest original MIDI note and preserve its velocity."""
    same_pitch_notes = [
        note
        for note in midi_notes
        if note.pitch == pitch
    ]

    if not same_pitch_notes:
        return 0

    best_note = min(
        same_pitch_notes,
        key=lambda item: (
            abs(item.start - start_seconds)
            + abs(item.end - end_seconds)
        ),
    )

    timing_difference = (
        abs(best_note.start - start_seconds)
        + abs(best_note.end - end_seconds)
    )

    # Basic Pitch MIDI events and note_events may differ slightly.
    if timing_difference <= 0.05:
        return best_note.velocity

    return 0


def collect_candidate_notes(
    midi_data: pretty_midi.PrettyMIDI,
    note_events: list,
    pitch_min: int,
    pitch_max: int,
    min_duration_seconds: float,
) -> list[CandidateNote]:
    """Read Basic Pitch candidates and attach confidence plus velocity."""
    midi_notes = collect_all_midi_notes(midi_data)

    candidates: list[CandidateNote] = []

    for event in note_events:
        if len(event) < 4:
            continue

        start_seconds = float(event[0])
        end_seconds = float(event[1])
        pitch = int(event[2])
        confidence = float(event[3])

        duration_seconds = end_seconds - start_seconds

        if duration_seconds < min_duration_seconds:
            continue

        if not pitch_min <= pitch <= pitch_max:
            continue

        velocity = find_matching_velocity(
            midi_notes=midi_notes,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            pitch=pitch,
        )

        if velocity <= 0:
            velocity = confidence_to_velocity(confidence)

        candidates.append(
            CandidateNote(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                pitch=pitch,
                velocity=velocity,
                confidence=confidence,
            )
        )

    return sorted(
        candidates,
        key=lambda item: (
            item.start_seconds,
            -item.confidence,
            item.pitch,
        ),
    )


def group_candidates_by_onset(
    candidates: list[CandidateNote],
    same_start_tolerance_seconds: float,
) -> list[list[CandidateNote]]:
    """Group notes that begin at approximately the same moment."""
    if not candidates:
        return []

    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            item.start_seconds,
            -item.confidence,
            item.pitch,
        ),
    )

    onset_groups: list[list[CandidateNote]] = []

    index = 0

    while index < len(sorted_candidates):
        anchor = sorted_candidates[index]

        group = [anchor]

        index += 1

        while (
            index < len(sorted_candidates)
            and abs(
                sorted_candidates[index].start_seconds
                - anchor.start_seconds
            ) <= same_start_tolerance_seconds
        ):
            group.append(sorted_candidates[index])
            index += 1

        onset_groups.append(group)

    return onset_groups


def make_raw_top_candidates(
    candidates: list[CandidateNote],
    max_candidates_per_onset: int,
    same_start_tolerance_seconds: float,
) -> list[RawNote]:
    """Keep the top confidence-ranked pitches per onset group.

    Rules:
    - No BPM conversion.
    - No beat grid.
    - No rhythm quantization.
    - Keep top 1 or top 2 confidence candidates per onset.
    - Current onset notes are shortened when the next onset begins.

    Rank 1:
        Highest confidence candidate.

    Rank 2:
        Second-highest confidence candidate.
    """
    if not candidates:
        return []

    onset_groups = group_candidates_by_onset(
        candidates=candidates,
        same_start_tolerance_seconds=same_start_tolerance_seconds,
    )

    final_notes: list[RawNote] = []

    for group_index, group in enumerate(onset_groups):
        ranked_group = sorted(
            group,
            key=lambda item: (
                item.confidence,
                item.velocity,
                item.end_seconds - item.start_seconds,
                item.pitch,
            ),
            reverse=True,
        )

        selected_candidates = ranked_group[
            :max_candidates_per_onset
        ]

        next_group_start: float | None = None

        if group_index < len(onset_groups) - 1:
            next_group_start = onset_groups[
                group_index + 1
            ][0].start_seconds

        for candidate_rank, candidate in enumerate(
            selected_candidates,
            start=1,
        ):
            final_end_seconds = candidate.end_seconds

            # End current detected event when the next onset begins.
            if (
                next_group_start is not None
                and final_end_seconds > next_group_start
            ):
                final_end_seconds = next_group_start

            if final_end_seconds <= candidate.start_seconds:
                continue

            final_notes.append(
                RawNote(
                    start_seconds=candidate.start_seconds,
                    end_seconds=final_end_seconds,
                    duration_seconds=(
                        final_end_seconds
                        - candidate.start_seconds
                    ),
                    pitch=candidate.pitch,
                    note_name=pretty_midi.note_number_to_name(
                        candidate.pitch
                    ),
                    velocity=candidate.velocity,
                    confidence=candidate.confidence,
                    candidate_rank=candidate_rank,
                )
            )

    return sorted(
        final_notes,
        key=lambda item: (
            item.start_seconds,
            item.candidate_rank,
            item.pitch,
        ),
    )


def write_raw_candidate_midi(
    output_path: Path,
    notes: list[RawNote],
) -> None:
    """Write raw candidate notes to MIDI.

    MIDI stores:
    - pitch
    - start time
    - end time
    - velocity

    MIDI does not natively store:
    - confidence
    - candidate rank

    Those are written to CSV and represented in PNG colors.
    """
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=120.0)

    instrument = pretty_midi.Instrument(
        program=0,
        is_drum=False,
        name="Raw Pitch Candidates",
    )

    for note in notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start_seconds,
                end=note.end_seconds,
            )
        )

    midi_data.instruments.append(instrument)
    midi_data.write(str(output_path))


def write_raw_notes_csv(
    output_path: Path,
    raw_notes: list[RawNote],
) -> None:
    """Write raw notes in column-oriented CSV form."""
    note_headers = [
        f"note_{index:03d}"
        for index in range(1, len(raw_notes) + 1)
    ]

    note_rows: list[dict[str, str | int]] = []

    for index, note in enumerate(raw_notes, start=1):
        note_rows.append(
            {
                "note_index": index,
                "note_name": note.note_name,
                "midi_pitch": note.pitch,
                "velocity": note.velocity,
                "confidence": f"{note.confidence:.6f}",
                "candidate_rank": note.candidate_rank,
                "start_seconds": f"{note.start_seconds:.6f}",
                "end_seconds": f"{note.end_seconds:.6f}",
                "duration_seconds": (
                    f"{note.duration_seconds:.6f}"
                ),
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
                        for row in note_rows
                    ],
                ]
            )


def build_event_times(
    raw_notes: list[RawNote],
) -> list[float]:
    """Create raw time columns from actual note starts and ends."""
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
    """Write an event-based piano roll CSV.

    Horizontal axis:
        Original note event times in seconds.

    Vertical axis:
        MIDI pitch.

    Cells:
        MIDI velocity when a note is active.

    This is not a BPM grid.
    """
    if not raw_notes:
        raise ValueError("No notes available for event-grid CSV.")

    min_pitch = max(0, min(note.pitch for note in raw_notes) - 2)
    max_pitch = min(127, max(note.pitch for note in raw_notes) + 2)

    event_times = build_event_times(raw_notes)

    time_headers = [
        f"{event_time:.6f}s"
        for event_time in event_times
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(["pitch_or_time", *time_headers])

        for pitch in range(max_pitch, min_pitch - 1, -1):
            row_cells: list[str] = []

            for event_time in event_times:
                active_notes = [
                    note
                    for note in raw_notes
                    if note.pitch == pitch
                    and note.start_seconds <= event_time
                    and event_time < note.end_seconds
                ]

                if active_notes:
                    highest_velocity = max(
                        note.velocity
                        for note in active_notes
                    )

                    row_cells.append(str(highest_velocity))
                else:
                    row_cells.append("")

            pitch_name = pretty_midi.note_number_to_name(pitch)

            writer.writerow(
                [
                    f"{pitch_name} (MIDI {pitch})",
                    *row_cells,
                ]
            )


def render_raw_piano_roll(
    output_path: Path,
    raw_notes: list[RawNote],
) -> None:
    """Render piano roll with rank-specific color behavior.

    Rank 1:
        Blue color based on MIDI velocity.
        Light blue = low velocity.
        Dark blue = high velocity.

    Rank 2:
        Fixed light purple.
        It intentionally does not use velocity color mapping.
    """
    if not raw_notes:
        raise ValueError("No notes available for piano-roll PNG.")

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

    # Rank 1 uses blue shades driven by MIDI velocity.
    velocity_normalizer = colors.Normalize(
        vmin=1,
        vmax=127,
    )

    velocity_blue_map = plt.cm.Blues

    # Rank 2 is always light purple.
    rank_2_color = "#B9A5E8"
    rank_2_border_color = "#6B4FA3"

    # Draw Rank 2 first, then draw Rank 1 above it.
    notes_for_draw = sorted(
        raw_notes,
        key=lambda item: (
            item.candidate_rank,
            item.start_seconds,
            item.pitch,
        ),
        reverse=True,
    )

    for note in notes_for_draw:
        if note.candidate_rank == 1:
            face_color = velocity_blue_map(
                velocity_normalizer(note.velocity)
            )

            edge_color = "#0D355F"
            alpha = 0.95

        else:
            face_color = rank_2_color
            edge_color = rank_2_border_color
            alpha = 0.82

        rectangle = Rectangle(
            (
                note.start_seconds,
                note.pitch - 0.36,
            ),
            note.duration_seconds,
            0.72,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=0.65,
            alpha=alpha,
        )

        ax.add_patch(rectangle)

    ax.set_xlim(-0.05, max_end_seconds + 0.15)
    ax.set_ylim(pitch_low - 0.6, pitch_high + 0.6)

    ax.set_xlabel("Original audio time (seconds)")
    ax.set_ylabel("Pitch")

    ax.set_title(
        "Raw Piano Roll | Rank 1: Blue by Velocity | "
        "Rank 2: Light Purple"
    )

    # Horizontal pitch guide lines.
    for pitch in range(pitch_low, pitch_high + 1):
        ax.axhline(
            pitch,
            color="#DEDEDE",
            linewidth=0.45,
            zorder=0,
        )

    # One-second vertical visual guides only.
    # This is not BPM quantization.
    second_count = int(math.ceil(max_end_seconds))

    for second in range(second_count + 1):
        ax.axvline(
            second,
            color="#D3DCE5",
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

    # Colorbar only applies to Rank 1 notes.
    velocity_colorbar_mapper = ScalarMappable(
        norm=velocity_normalizer,
        cmap=velocity_blue_map,
    )

    velocity_colorbar_mapper.set_array([])

    colorbar = fig.colorbar(
        velocity_colorbar_mapper,
        ax=ax,
        pad=0.015,
    )

    colorbar.set_label(
        "Rank 1 MIDI velocity (light blue -> dark blue)",
        rotation=270,
        labelpad=18,
    )

    legend_rank_1_low = Rectangle(
        (0, 0),
        1,
        1,
        facecolor=velocity_blue_map(
            velocity_normalizer(30)
        ),
        edgecolor="#0D355F",
        alpha=0.95,
        label="Rank 1: low velocity",
    )

    legend_rank_1_high = Rectangle(
        (0, 0),
        1,
        1,
        facecolor=velocity_blue_map(
            velocity_normalizer(110)
        ),
        edgecolor="#0D355F",
        alpha=0.95,
        label="Rank 1: high velocity",
    )

    legend_rank_2 = Rectangle(
        (0, 0),
        1,
        1,
        facecolor=rank_2_color,
        edgecolor=rank_2_border_color,
        alpha=0.82,
        label="Rank 2: second confidence candidate",
    )

    ax.legend(
        handles=[
            legend_rank_1_low,
            legend_rank_1_high,
            legend_rank_2,
        ],
        loc="upper right",
        framealpha=0.94,
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
    max_candidates_per_onset: int,
) -> dict[str, dict[str, Path]]:
    """Create output paths for one source audio file."""
    source_name = source_path.stem
    mode_label = f"top{max_candidates_per_onset}"

    source_dir = output_root / "by_source" / source_name

    source_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate_dirs = {
        key: output_root / folder_name
        for key, folder_name in OUTPUT_FOLDERS.items()
    }

    for directory in aggregate_dirs.values():
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    source_paths = {
        "basic_pitch_original_midi": (
            source_dir
            / f"{source_name}_01_basic_pitch_original.mid"
        ),
        "raw_candidates_midi": (
            source_dir
            / f"{source_name}_02_raw_{mode_label}_candidates.mid"
        ),
        "raw_notes_csv": (
            source_dir
            / f"{source_name}_03_raw_{mode_label}_notes.csv"
        ),
        "raw_pianoroll_png": (
            source_dir
            / f"{source_name}_04_raw_{mode_label}_pianoroll.png"
        ),
        "raw_pianoroll_event_grid_csv": (
            source_dir
            / f"{source_name}_05_raw_{mode_label}_event_grid.csv"
        ),
    }

    aggregate_paths = {
        "basic_pitch_original_midi": (
            aggregate_dirs["basic_pitch_original_midi"]
            / f"{source_name}_01_basic_pitch_original.mid"
        ),
        "raw_candidates_midi": (
            aggregate_dirs["raw_candidates_midi"]
            / f"{source_name}_02_raw_{mode_label}_candidates.mid"
        ),
        "raw_notes_csv": (
            aggregate_dirs["raw_notes_csv"]
            / f"{source_name}_03_raw_{mode_label}_notes.csv"
        ),
        "raw_pianoroll_png": (
            aggregate_dirs["raw_pianoroll_png"]
            / f"{source_name}_04_raw_{mode_label}_pianoroll.png"
        ),
        "raw_pianoroll_event_grid_csv": (
            aggregate_dirs["raw_pianoroll_event_grid_csv"]
            / f"{source_name}_05_raw_{mode_label}_event_grid.csv"
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
    """Copy source-level files into aggregate type folders."""
    for key, source_file in source_paths.items():
        shutil.copy2(
            source_file,
            aggregate_paths[key],
        )


def process_one_file(
    source_path: Path,
    output_root: Path,
    max_candidates_per_onset: int,
    same_start_tolerance_ms: float,
    min_duration_ms: float,
    onset_threshold: float,
    frame_threshold: float,
    pitch_min: int,
    pitch_max: int,
) -> int:
    """Process one MP3 file."""
    basic_pitch_midi, note_events = get_basic_pitch_result(
        source_path=source_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length_ms=min_duration_ms,
    )

    print("  [STEP 2/5] Reading confidence-ranked note candidates")

    candidate_notes = collect_candidate_notes(
        midi_data=basic_pitch_midi,
        note_events=note_events,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        min_duration_seconds=min_duration_ms / 1000.0,
    )

    if not candidate_notes:
        raise ValueError(
            "No usable note candidates were detected."
        )

    print(
        "  [INFO] Keeping top "
        f"{max_candidates_per_onset} confidence candidate(s) "
        "per onset group."
    )

    raw_notes = make_raw_top_candidates(
        candidates=candidate_notes,
        max_candidates_per_onset=max_candidates_per_onset,
        same_start_tolerance_seconds=(
            same_start_tolerance_ms / 1000.0
        ),
    )

    if not raw_notes:
        raise ValueError(
            "No notes remain after candidate filtering."
        )

    output_paths = build_output_paths(
        output_root=output_root,
        source_path=source_path,
        max_candidates_per_onset=max_candidates_per_onset,
    )

    print("  [STEP 3/5] Writing MIDI files")

    # Basic Pitch original MIDI, before top-1/top-2 filtering.
    basic_pitch_midi.write(
        str(
            output_paths["source"][
                "basic_pitch_original_midi"
            ]
        )
    )

    # Filtered output MIDI.
    write_raw_candidate_midi(
        output_path=(
            output_paths["source"][
                "raw_candidates_midi"
            ]
        ),
        notes=raw_notes,
    )

    print("  [STEP 4/5] Writing CSV files")

    write_raw_notes_csv(
        output_path=(
            output_paths["source"]["raw_notes_csv"]
        ),
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

    print("  [STEP 5/5] Rendering PNG piano roll")

    render_raw_piano_roll(
        output_path=(
            output_paths["source"][
                "raw_pianoroll_png"
            ]
        ),
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

    print("=" * 72)
    print("[INFO] MP3 -> Raw Top Confidence Candidates")
    print("[INFO] No BPM conversion.")
    print("[INFO] No beat grid.")
    print("[INFO] No rhythm quantization.")
    print(f"[INFO] Input : {args.input_dir.resolve()}")
    print(f"[INFO] Output: {args.output_dir.resolve()}")
    print(
        "[INFO] Candidates per onset: "
        f"{args.max_candidates_per_onset}"
    )
    print(
        "[INFO] Same-start tolerance: "
        f"{args.same_start_tolerance_ms:g} ms"
    )
    print(
        "[INFO] Minimum duration: "
        f"{args.min_duration_ms:g} ms"
    )
    print(
        "[INFO] Pitch range: "
        f"MIDI {args.pitch_min}-{args.pitch_max}"
    )
    print("[INFO] PNG display:")
    print("[INFO]   Rank 1 = blue, darker means higher velocity.")
    print("[INFO]   Rank 2 = fixed light purple.")
    print("=" * 72)

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
                max_candidates_per_onset=(
                    args.max_candidates_per_onset
                ),
                same_start_tolerance_ms=(
                    args.same_start_tolerance_ms
                ),
                min_duration_ms=args.min_duration_ms,
                onset_threshold=args.onset_threshold,
                frame_threshold=args.frame_threshold,
                pitch_min=args.pitch_min,
                pitch_max=args.pitch_max,
            )

            print(
                f"  [OK] Done. Final notes: {note_count}"
            )

            success_count += 1

        except Exception as error:
            print(f"  [FAIL] {error}")

    print()
    print("=" * 72)
    print(
        "[DONE] Successfully processed: "
        f"{success_count}/{len(mp3_files)}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
