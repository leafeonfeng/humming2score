from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from dataclasses import dataclass, replace
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
    "raw_guitar_tab_preview_png": "06_raw_guitar_tab_preview_png",
    "raw_guitar_tab_csv": "07_raw_guitar_tab_csv",
    "raw_guitar_tab_with_noteheads_png": "08_raw_guitar_tab_with_noteheads_png", 
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
    "merged_segment_count",
]

# Standard guitar tuning, listed from the lowest string to the highest.
# Index 0 is the low E string, drawn at the bottom of the tab.
GUITAR_STRINGS = [
    ("E", 40),
    ("A", 45),
    ("D", 50),
    ("G", 55),
    ("B", 59),
    ("e", 64),
]

GUITAR_MAX_FRET = 24

# Snap targets expressed in eighth-note units.
DURATION_UNIT_SNAP_TARGETS = [1, 2, 3, 4, 6, 8, 12, 16]


@dataclass
class CandidateNote:
    start_seconds: float
    end_seconds: float
    pitch: int
    velocity: int
    confidence: float


@dataclass
class RawNote:
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    pitch: int
    note_name: str
    velocity: int
    confidence: float
    candidate_rank: int
    detected_end_seconds: float = 0.0
    last_segment_velocity: int = 0
    merged_segment_count: int = 1


@dataclass
class GuitarPosition:
    string_index: int
    string_name: str
    open_pitch: int
    fret: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract raw Basic Pitch candidates from MP3 files and export "
            "piano-roll plus guitar-tab drafts. No BPM detection and no key "
            "transposition."
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
            "Number of confidence-ranked pitches to keep per onset group. "
            "Default: 2"
        ),
    )

    parser.add_argument(
        "--same-start-tolerance-ms",
        type=float,
        default=15.0,
        help=(
            "Notes starting within this time are treated as the same onset. "
            "Default: 15"
        ),
    )

    parser.add_argument(
        "--min-duration-ms",
        type=float,
        default=45.0,
        help=(
            "Discard detected notes shorter than this duration. Default: 45"
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

    parser.add_argument(
        "--merge-same-pitch",
        dest="merge_same_pitch",
        action="store_true",
        default=True,
        help=(
            "Merge consecutive same-pitch segments that look like one "
            "sustained note. Enabled by default."
        ),
    )

    parser.add_argument(
        "--no-merge-same-pitch",
        dest="merge_same_pitch",
        action="store_false",
        help="Keep every detected segment as a separate note.",
    )

    parser.add_argument(
        "--merge-max-gap-ms",
        type=float,
        default=60.0,
        help=(
            "Maximum silence between two same-pitch segments that still "
            "counts as one sustained note. Default: 60"
        ),
    )

    parser.add_argument(
        "--merge-velocity-jump",
        type=float,
        default=10.0,
        help=(
            "If the next same-pitch segment is louder than the previous one "
            "by more than this MIDI velocity amount, treat it as a new "
            "attack instead of a sustain. Default: 10"
        ),
    )

    parser.add_argument(
        "--png-note-width-scale",
        type=float,
        default=1.6,
        help=(
            "Visual width multiplier for piano-roll blocks. This does not "
            "modify MIDI or CSV durations. Default: 1.6"
        ),
    )

    parser.add_argument(
        "--png-min-block-ms",
        type=float,
        default=90.0,
        help=(
            "Minimum visual block width in the piano-roll PNG, in "
            "milliseconds. This does not modify note data. Default: 90"
        ),
    )

    parser.add_argument(
        "--duration-reference",
        choices=["percentile", "shortest"],
        default="percentile",
        help=(
            "Reference duration for one eighth-note unit. 'percentile' "
            "ignores extreme short notes. 'shortest' uses the absolute "
            "shortest note. Default: percentile"
        ),
    )

    parser.add_argument(
        "--duration-percentile",
        type=float,
        default=10.0,
        help=(
            "Percentile used when --duration-reference=percentile. "
            "Default: 10"
        ),
    )

    parser.add_argument(
        "--snap-duration-units",
        dest="snap_duration_units",
        action="store_true",
        default=True,
        help=(
            "Snap durations to common note values such as 1, 2, 3, 4, 6, 8 "
            "eighth-note units. Enabled by default."
        ),
    )

    parser.add_argument(
        "--no-snap-duration-units",
        dest="snap_duration_units",
        action="store_false",
        help="Keep the raw rounded eighth-note unit count.",
    )

    parser.add_argument(
        "--tab-events-per-system",
        type=int,
        default=16,
        help=(
            "Maximum number of onset groups on each guitar-tab system. "
            "Default: 16"
        ),
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

    if args.merge_max_gap_ms < 0:
        parser.error("--merge-max-gap-ms cannot be negative.")

    if args.merge_velocity_jump < 0:
        parser.error("--merge-velocity-jump cannot be negative.")

    if args.png_note_width_scale <= 0:
        parser.error("--png-note-width-scale must be greater than 0.")

    if args.png_min_block_ms < 0:
        parser.error("--png-min-block-ms cannot be negative.")

    if not 0 < args.duration_percentile <= 100:
        parser.error("--duration-percentile must be in (0, 100].")

    if args.tab_events_per_system < 4:
        parser.error("--tab-events-per-system must be at least 4.")

    return args


def collect_mp3_files(input_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def get_basic_pitch_result(
    source_path: Path,
    onset_threshold: float,
    frame_threshold: float,
    minimum_note_length_ms: float,
) -> tuple[pretty_midi.PrettyMIDI, list]:
    print(f"  [STEP 1/7] Basic Pitch transcription: {source_path.name}")

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
    notes: list[pretty_midi.Note] = []

    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)

    return sorted(
        notes,
        key=lambda item: (item.start, item.end, item.pitch),
    )


def confidence_to_velocity(confidence: float) -> int:
    confidence = max(0.0, min(1.0, confidence))
    return max(1, min(127, round(1 + confidence * 126)))


def find_matching_velocity(
    midi_notes: list[pretty_midi.Note],
    start_seconds: float,
    end_seconds: float,
    pitch: int,
) -> int:
    same_pitch_notes = [note for note in midi_notes if note.pitch == pitch]

    if not same_pitch_notes:
        return 0

    best_note = min(
        same_pitch_notes,
        key=lambda item: (
            abs(item.start - start_seconds) + abs(item.end - end_seconds)
        ),
    )

    timing_difference = abs(best_note.start - start_seconds) + abs(
        best_note.end - end_seconds
    )

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
    midi_notes = collect_all_midi_notes(midi_data)
    candidates: list[CandidateNote] = []

    for event in note_events:
        if len(event) < 4:
            continue

        start_seconds = float(event[0])
        end_seconds = float(event[1])
        pitch = int(event[2])
        confidence = float(event[3])

        if end_seconds - start_seconds < min_duration_seconds:
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
            and abs(sorted_candidates[index].start_seconds - anchor.start_seconds)
            <= same_start_tolerance_seconds
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

        selected_candidates = ranked_group[:max_candidates_per_onset]
        next_group_start: float | None = None

        if group_index < len(onset_groups) - 1:
            next_group_start = onset_groups[group_index + 1][0].start_seconds

        for candidate_rank, candidate in enumerate(selected_candidates, start=1):
            final_end_seconds = candidate.end_seconds

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
                    duration_seconds=(final_end_seconds - candidate.start_seconds),
                    pitch=candidate.pitch,
                    note_name=pretty_midi.note_number_to_name(candidate.pitch),
                    velocity=candidate.velocity,
                    confidence=candidate.confidence,
                    candidate_rank=candidate_rank,
                    detected_end_seconds=candidate.end_seconds,
                    last_segment_velocity=candidate.velocity,
                    merged_segment_count=1,
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


def looks_like_one_sustained_note(
    previous_note: RawNote,
    next_note: RawNote,
    max_gap_seconds: float,
    velocity_jump_threshold: float,
) -> bool:
    if previous_note.pitch != next_note.pitch:
        return False

    if previous_note.candidate_rank != next_note.candidate_rank:
        return False

    gap_seconds = next_note.start_seconds - previous_note.detected_end_seconds

    if gap_seconds > max_gap_seconds:
        return False

    velocity_jump = next_note.velocity - previous_note.last_segment_velocity

    if velocity_jump > velocity_jump_threshold:
        return False

    return True


def merge_sustained_same_pitch_notes(
    raw_notes: list[RawNote],
    max_gap_seconds: float,
    velocity_jump_threshold: float,
) -> list[RawNote]:
    if not raw_notes:
        return []

    ordered_notes = sorted(
        raw_notes,
        key=lambda note: (
            note.candidate_rank,
            note.start_seconds,
            note.pitch,
        ),
    )

    streams: dict[int, list[RawNote]] = {}

    for note in ordered_notes:
        stream = streams.setdefault(note.candidate_rank, [])

        if stream:
            previous_note = stream[-1]

            if looks_like_one_sustained_note(
                previous_note=previous_note,
                next_note=note,
                max_gap_seconds=max_gap_seconds,
                velocity_jump_threshold=velocity_jump_threshold,
            ):
                merged_end_seconds = max(
                    previous_note.end_seconds,
                    note.end_seconds,
                )

                stream[-1] = replace(
                    previous_note,
                    end_seconds=merged_end_seconds,
                    duration_seconds=(
                        merged_end_seconds - previous_note.start_seconds
                    ),
                    velocity=max(
                        previous_note.velocity,
                        note.velocity,
                    ),
                    confidence=max(
                        previous_note.confidence,
                        note.confidence,
                    ),
                    detected_end_seconds=max(
                        previous_note.detected_end_seconds,
                        note.detected_end_seconds,
                    ),
                    last_segment_velocity=note.velocity,
                    merged_segment_count=(
                        previous_note.merged_segment_count + 1
                    ),
                )
                continue

        stream.append(replace(note))

    merged_notes: list[RawNote] = []

    for stream in streams.values():
        merged_notes.extend(stream)

    return sorted(
        merged_notes,
        key=lambda note: (
            note.start_seconds,
            note.candidate_rank,
            note.pitch,
        ),
    )


def write_raw_candidate_midi(
    output_path: Path,
    notes: list[RawNote],
) -> None:
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
    note_headers = [f"note_{index:03d}" for index in range(1, len(raw_notes) + 1)]

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
                "duration_seconds": f"{note.duration_seconds:.6f}",
                "merged_segment_count": note.merged_segment_count,
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
                    *[row.get(field_name, "") for row in note_rows],
                ]
            )


def build_event_times(raw_notes: list[RawNote]) -> list[float]:
    event_times = {round(note.start_seconds, 6) for note in raw_notes}
    event_times.update(round(note.end_seconds, 6) for note in raw_notes)
    return sorted(event_times)


def write_raw_pianoroll_event_grid_csv(
    output_path: Path,
    raw_notes: list[RawNote],
) -> None:
    if not raw_notes:
        raise ValueError("No notes available for event-grid CSV.")

    min_pitch = max(0, min(note.pitch for note in raw_notes) - 2)
    max_pitch = min(127, max(note.pitch for note in raw_notes) + 2)

    event_times = build_event_times(raw_notes)
    time_headers = [f"{event_time:.6f}s" for event_time in event_times]

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
                    row_cells.append(
                        str(max(note.velocity for note in active_notes))
                    )
                else:
                    row_cells.append("")

            pitch_name = pretty_midi.note_number_to_name(pitch)
            writer.writerow([f"{pitch_name} (MIDI {pitch})", *row_cells])


def render_raw_piano_roll(
    output_path: Path,
    raw_notes: list[RawNote],
    note_width_scale: float,
    minimum_visual_width_seconds: float,
) -> None:
    if not raw_notes:
        raise ValueError("No notes available for piano-roll PNG.")

    min_pitch = min(note.pitch for note in raw_notes)
    max_pitch = max(note.pitch for note in raw_notes)

    pitch_low = max(0, min_pitch - 2)
    pitch_high = min(127, max_pitch + 2)

    def visual_width(note: RawNote) -> float:
        return max(
            note.duration_seconds * note_width_scale,
            minimum_visual_width_seconds,
        )

    max_visual_end_seconds = max(
        note.start_seconds + visual_width(note) for note in raw_notes
    )

    figure_width = max(12, min(48, 9 + max_visual_end_seconds * 0.72))
    figure_height = max(4, min(13, 3 + (pitch_high - pitch_low) * 0.34))

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    velocity_normalizer = colors.Normalize(vmin=1, vmax=127)
    velocity_blue_map = plt.cm.Blues

    rank_2_color = "#B9A5E8"
    rank_2_border_color = "#6B4FA3"

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
            face_color = velocity_blue_map(velocity_normalizer(note.velocity))
            edge_color = "#0D355F"
            alpha = 0.95
        else:
            face_color = rank_2_color
            edge_color = rank_2_border_color
            alpha = 0.82

        ax.add_patch(
            Rectangle(
                (note.start_seconds, note.pitch - 0.36),
                visual_width(note),
                0.72,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=0.65,
                alpha=alpha,
            )
        )

    ax.set_xlim(-0.05, max_visual_end_seconds + 0.15)
    ax.set_ylim(pitch_low - 0.6, pitch_high + 0.6)

    ax.set_xlabel("Original audio time (seconds)")
    ax.set_ylabel("Pitch")

    ax.set_title(
        "Raw Piano Roll | Display width only: "
        f"x{note_width_scale:g} | "
        "Rank 1: Blue | Rank 2: Light Purple"
    )

    for pitch in range(pitch_low, pitch_high + 1):
        ax.axhline(pitch, color="#DEDEDE", linewidth=0.45, zorder=0)

    for second in range(int(math.ceil(max_visual_end_seconds)) + 1):
        ax.axvline(second, color="#D3DCE5", linewidth=0.7, zorder=0)

    pitch_ticks = list(range(pitch_low, pitch_high + 1))

    ax.set_yticks(pitch_ticks)
    ax.set_yticklabels(
        [pretty_midi.note_number_to_name(pitch) for pitch in pitch_ticks]
    )

    colorbar_mapper = ScalarMappable(
        norm=velocity_normalizer,
        cmap=velocity_blue_map,
    )
    colorbar_mapper.set_array([])

    colorbar = fig.colorbar(colorbar_mapper, ax=ax, pad=0.015)
    colorbar.set_label(
        "Rank 1 MIDI velocity (light blue -> dark blue)",
        rotation=270,
        labelpad=18,
    )

    legend_rank_1_low = Rectangle(
        (0, 0),
        1,
        1,
        facecolor=velocity_blue_map(velocity_normalizer(30)),
        edgecolor="#0D355F",
        alpha=0.95,
        label="Rank 1: low velocity",
    )

    legend_rank_1_high = Rectangle(
        (0, 0),
        1,
        1,
        facecolor=velocity_blue_map(velocity_normalizer(110)),
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

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def percentile_value(
    values: list[float],
    percentile: float,
) -> float:
    if not values:
        raise ValueError("Cannot calculate percentile of empty values.")

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    position = (len(ordered) - 1) * percentile / 100.0
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))

    if lower_index == upper_index:
        return float(ordered[lower_index])

    fraction = position - lower_index

    return float(
        ordered[lower_index]
        + (ordered[upper_index] - ordered[lower_index]) * fraction
    )


def calculate_duration_reference(
        raw_notes: list[RawNote],
        reference_mode: str,
        percentile: float,
) -> float:
    durations = [
        note.duration_seconds
        for note in raw_notes
        if note.duration_seconds > 0
    ]

    if not durations:
        raise ValueError(
            "Cannot calculate a duration reference without notes."
        )

    # 直接取最短的那个音符作为"1个八分音符"
    # 不管 mode 和 percentile
    reference = min(durations)

    print(f"  [INFO] Shortest duration: {reference:.4f}s, using as 1 eighth-note unit")

    return max(reference, 0.001)


def snap_duration_units(units: int) -> int:
    if units in DURATION_UNIT_SNAP_TARGETS:
        return units

    if units > DURATION_UNIT_SNAP_TARGETS[-1]:
        return int(round(units / 4.0)) * 4

    return min(
        DURATION_UNIT_SNAP_TARGETS,
        key=lambda target: (abs(target - units), target),
    )


def duration_units_from_seconds(
    duration_seconds: float,
    reference_seconds: float,
    use_snap: bool,
) -> int:
    if reference_seconds <= 0:
        return 1

    units = max(1, int(round(duration_seconds / reference_seconds)))

    if use_snap:
        units = max(1, snap_duration_units(units))

    return units


def duration_name_from_units(units: int) -> str:
    names = {
        1: "eighth",
        2: "quarter",
        3: "dotted_quarter",
        4: "half",
        6: "dotted_half",
        8: "whole",
        12: "dotted_whole",
        16: "double_whole",
    }

    return names.get(units, f"{units}_eighths")


def choose_guitar_position(midi_pitch: int) -> GuitarPosition | None:
    possible_positions: list[GuitarPosition] = []

    for string_index, (string_name, open_pitch) in enumerate(GUITAR_STRINGS):
        fret = midi_pitch - open_pitch

        if 0 <= fret <= GUITAR_MAX_FRET:
            possible_positions.append(
                GuitarPosition(
                    string_index=string_index,
                    string_name=string_name,
                    open_pitch=open_pitch,
                    fret=fret,
                )
            )

    if not possible_positions:
        return None

    return min(
        possible_positions,
        key=lambda position: (position.fret, position.string_index),
    )


def group_raw_notes_for_tab(
    raw_notes: list[RawNote],
    same_start_tolerance_seconds: float,
) -> list[list[RawNote]]:
    sorted_notes = sorted(
        raw_notes,
        key=lambda note: (
            note.start_seconds,
            note.candidate_rank,
            note.pitch,
        ),
    )

    groups: list[list[RawNote]] = []
    index = 0

    while index < len(sorted_notes):
        anchor_start = sorted_notes[index].start_seconds
        group: list[RawNote] = []

        while (
            index < len(sorted_notes)
            and abs(sorted_notes[index].start_seconds - anchor_start)
            <= same_start_tolerance_seconds
        ):
            group.append(sorted_notes[index])
            index += 1

        groups.append(group)

    return groups


def pick_group_duration_note(group: list[RawNote]) -> RawNote:
    rank_1_notes = [note for note in group if note.candidate_rank == 1]

    if rank_1_notes:
        return rank_1_notes[0]

    return group[0]


def write_guitar_tab_csv(
    output_path: Path,
    raw_notes: list[RawNote],
    same_start_tolerance_seconds: float,
    reference_seconds: float,
    reference_mode: str,
    use_snap: bool,
) -> None:
    groups = group_raw_notes_for_tab(
        raw_notes=raw_notes,
        same_start_tolerance_seconds=same_start_tolerance_seconds,
    )

    headers = [
        "onset_index",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "duration_reference_seconds",
        "duration_reference_mode",
        "note_name",
        "midi_pitch",
        "velocity",
        "confidence",
        "candidate_rank",
        "guitar_string",
        "string_number_from_high_e",
        "open_string_note",
        "fret",
        "duration_units",
        "duration_name",
        "merged_segment_count",
        "playable_on_standard_guitar",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        for onset_index, group in enumerate(groups, start=1):
            for note in group:
                position = choose_guitar_position(note.pitch)

                duration_units = duration_units_from_seconds(
                    duration_seconds=note.duration_seconds,
                    reference_seconds=reference_seconds,
                    use_snap=use_snap,
                )

                if position is None:
                    guitar_string = ""
                    string_number = ""
                    open_string_note = ""
                    fret: int | str = ""
                    playable = 0
                else:
                    guitar_string = position.string_name
                    string_number = 6 - position.string_index
                    open_string_note = pretty_midi.note_number_to_name(
                        position.open_pitch
                    )
                    fret = position.fret
                    playable = 1

                writer.writerow(
                    {
                        "onset_index": onset_index,
                        "start_seconds": f"{note.start_seconds:.6f}",
                        "end_seconds": f"{note.end_seconds:.6f}",
                        "duration_seconds": f"{note.duration_seconds:.6f}",
                        "duration_reference_seconds": f"{reference_seconds:.6f}",
                        "duration_reference_mode": reference_mode,
                        "note_name": note.note_name,
                        "midi_pitch": note.pitch,
                        "velocity": note.velocity,
                        "confidence": f"{note.confidence:.6f}",
                        "candidate_rank": note.candidate_rank,
                        "guitar_string": guitar_string,
                        "string_number_from_high_e": string_number,
                        "open_string_note": open_string_note,
                        "fret": fret,
                        "duration_units": duration_units,
                        "duration_name": duration_name_from_units(duration_units),
                        "merged_segment_count": note.merged_segment_count,
                        "playable_on_standard_guitar": playable,
                    }
                )


def render_guitar_tab_preview(
    output_path: Path,
    raw_notes: list[RawNote],
    same_start_tolerance_seconds: float,
    reference_seconds: float,
    reference_mode: str,
    use_snap: bool,
    events_per_system: int,
) -> None:
    if not raw_notes:
        raise ValueError("No notes available for guitar-tab preview.")

    groups = group_raw_notes_for_tab(
        raw_notes=raw_notes,
        same_start_tolerance_seconds=same_start_tolerance_seconds,
    )

    group_units = [
        duration_units_from_seconds(
            duration_seconds=pick_group_duration_note(group).duration_seconds,
            reference_seconds=reference_seconds,
            use_snap=use_snap,
        )
        for group in groups
    ]

    systems: list[list[int]] = []

    for start_index in range(0, len(groups), events_per_system):
        systems.append(
            list(
                range(
                    start_index,
                    min(len(groups), start_index + events_per_system),
                )
            )
        )

    system_total_units = [
        sum(group_units[index] for index in system) for system in systems
    ]

    max_total_units = max(system_total_units)
    system_count = len(systems)

    figure_width = max(13.0, min(60.0, 3.0 + max_total_units * 0.45))
    figure_height = max(5.0, system_count * 3.1)

    fig, axes = plt.subplots(
        system_count,
        1,
        figsize=(figure_width, figure_height),
        squeeze=False,
    )

    axes_list = [axes[index][0] for index in range(system_count)]
    x_start = 1.0
    string_label_x = x_start - 0.6

    for system_index, ax in enumerate(axes_list):
        group_indexes = systems[system_index]
        system_units = system_total_units[system_index]
        line_end_x = x_start + max_total_units + 0.4

        for unit_index in range(max_total_units + 1):
            grid_x = x_start + unit_index
            is_group_of_eight = unit_index % 8 == 0

            ax.axvline(
                grid_x,
                color="#C9D4DF" if is_group_of_eight else "#E8EDF2",
                linewidth=0.85 if is_group_of_eight else 0.55,
                zorder=0,
            )

        for string_index, (string_name, open_pitch) in enumerate(GUITAR_STRINGS):
            ax.plot(
                [x_start - 0.25, line_end_x],
                [string_index, string_index],
                color="#222222",
                linewidth=0.9,
                zorder=1,
            )

            ax.text(
                string_label_x,
                string_index,
                string_name,
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=3,
            )

            ax.text(
                string_label_x - 0.55,
                string_index,
                pretty_midi.note_number_to_name(open_pitch),
                fontsize=7,
                color="#888888",
                ha="center",
                va="center",
                zorder=3,
            )

        cursor_units = 0

        for local_index, group_index in enumerate(group_indexes, start=1):
            group = groups[group_index]
            units = group_units[group_index]
            x = x_start + cursor_units + 0.5
            cursor_units += units

            notes_for_draw = sorted(
                group,
                key=lambda note: note.candidate_rank,
                reverse=True,
            )

            drawn_rows: set[int] = set()

            for note in notes_for_draw:
                position = choose_guitar_position(note.pitch)

                if position is None:
                    continue

                visual_row = position.string_index
                note_x = x

                if note.candidate_rank == 2 and visual_row in drawn_rows:
                    note_x = x - 0.30

                if note.candidate_rank == 1:
                    text = str(position.fret)
                    text_color = "#173F6F"
                    edge_color = "#0D2745"
                    font_size = 11
                    font_weight = "bold"
                    zorder = 6
                else:
                    text = f"({position.fret})"
                    text_color = "#684B9B"
                    edge_color = "#B9A5E8"
                    font_size = 9
                    font_weight = "normal"
                    zorder = 5

                note_units = duration_units_from_seconds(
                    duration_seconds=note.duration_seconds,
                    reference_seconds=reference_seconds,
                    use_snap=use_snap,
                )

                if note_units >= 2:
                    dash_start_x = note_x + 0.30
                    dash_end_x = x + note_units - 0.35

                    if dash_end_x > dash_start_x:
                        ax.plot(
                            [dash_start_x, dash_end_x],
                            [visual_row, visual_row],
                            color=(
                                "#173F6F"
                                if note.candidate_rank == 1
                                else "#B9A5E8"
                            ),
                            linewidth=(2.2 if note.candidate_rank == 1 else 1.5),
                            solid_capstyle="butt",
                            zorder=zorder - 1,
                        )

                ax.text(
                    note_x,
                    visual_row,
                    text,
                    fontsize=font_size,
                    color=text_color,
                    fontweight=font_weight,
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.12",
                        "facecolor": "white",
                        "edgecolor": edge_color,
                        "linewidth": 0.65,
                        "alpha": 0.95,
                    },
                    zorder=zorder,
                )

                drawn_rows.add(visual_row)

            duration_note = pick_group_duration_note(group)

            ax.text(
                x,
                -0.62,
                str(units),
                fontsize=8,
                color="#555555",
                ha="center",
                va="top",
            )

            if (
                local_index == 1
                or local_index % 4 == 0
                or group_index == len(groups) - 1
            ):
                ax.text(
                    x,
                    -1.12,
                    f"{duration_note.start_seconds:.2f}s",
                    fontsize=7,
                    color="#888888",
                    ha="center",
                    va="top",
                )

        ax.set_xlim(string_label_x - 1.1, x_start + max_total_units + 0.8)
        ax.set_ylim(-1.75, 5.6)
        ax.set_xticks([])
        ax.set_yticks([])

        system_units = system_total_units[system_index]
      
        ax.set_title(
            f"System {system_index + 1} | "
            f"{len(group_indexes)} onsets | "
            f"{system_units} eighth-note units",
            loc="left",
            fontsize=9,
            color="#444444",
            pad=6,
        )

        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Raw Guitar Tab Draft | Standard Tuning, low E at bottom",
        fontsize=15,
        y=0.995,
    )

    fig.text(
        0.5,
        0.012,
        (
            "One unit = one eighth note = "
            f"{reference_seconds:.3f}s ({reference_mode} reference). "
            "Numbers under the tab are unit counts; a dash means the note "
            "is held. Rank 1 is the main candidate, purple parentheses are "
            "the alternative. Absolute pitch, no key signature. The heavier "
            "line every 8 units is a counting aid, not a detected bar."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout(rect=(0, 0.05, 1, 0.965))

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def draw_notehead(
        ax,
        x: float,
        y: float,
        units: int,
        is_rank_1: bool,
) -> None:
    """Draw a standard music notation notehead at the given position."""

    # 判断基础音符类型（忽略附点）
    base_units = units
    has_dot = False

    if units == 3:  # 附点四分音符
        base_units = 2
        has_dot = True
    elif units == 6:  # 附点二分音符
        base_units = 4
        has_dot = True
    elif units == 12:  # 附点全音符
        base_units = 8
        has_dot = True

    # 根据基础音符类型决定样式
    is_filled = base_units <= 2  # 八分音符(1) 和 四分音符(2) 是实心
    has_stem = base_units < 8  # 全音符(8) 没有符干
    has_flag = base_units == 1  # 只有八分音符有符尾

    # Colors
    if is_rank_1:
        fill_color = "#173F6F" if is_filled else "white"
        edge_color = "#0D2745"
        stem_color = "#0D2745"
    else:
        fill_color = "#684B9B" if is_filled else "white"
        edge_color = "#B9A5E8"
        stem_color = "#B9A5E8"

    # Draw ellipse (notehead)
    from matplotlib.patches import Ellipse

    ellipse = Ellipse(
        (x, y),
        width=0.22,
        height=0.15,
        angle=-20,
        facecolor=fill_color,
        edgecolor=edge_color,
        linewidth=1.2,
        zorder=10,
    )
    ax.add_patch(ellipse)

    # Draw stem
    if has_stem:
        stem_height = 0.75
        stem_x = x + 0.10
        stem_top_y = y + stem_height

        ax.plot(
            [stem_x, stem_x],
            [y, stem_top_y],
            color=stem_color,
            linewidth=1.8,
            solid_capstyle="butt",
            zorder=9,
        )

        # Draw flag for eighth note
        if has_flag:
            flag_x = [stem_x, stem_x + 0.15, stem_x + 0.18]
            flag_y = [stem_top_y, stem_top_y - 0.25, stem_top_y - 0.35]
            ax.plot(
                flag_x,
                flag_y,
                color=stem_color,
                linewidth=2.0,
                solid_capstyle="round",
                zorder=9,
            )

    # Draw dot for dotted notes
    if has_dot:
        dot_x = x + 0.28
        ax.plot(
            dot_x,
            y,
            marker="o",
            markersize=3.5,
            color=edge_color,
            zorder=11,
        )


def render_guitar_tab_with_noteheads(
    output_path: Path,
    raw_notes: list[RawNote],
    same_start_tolerance_seconds: float,
    reference_seconds: float,
    reference_mode: str,
    use_snap: bool,
    events_per_system: int,
) -> None:
    """Render guitar tab with standard music notation noteheads."""
    
    if not raw_notes:
        raise ValueError("No notes available for guitar-tab with noteheads.")

    groups = group_raw_notes_for_tab(
        raw_notes=raw_notes,
        same_start_tolerance_seconds=same_start_tolerance_seconds,
    )

    group_units = [
        duration_units_from_seconds(
            duration_seconds=pick_group_duration_note(group).duration_seconds,
            reference_seconds=reference_seconds,
            use_snap=use_snap,
        )
        for group in groups
    ]

    systems: list[list[int]] = []

    for start_index in range(0, len(groups), events_per_system):
        systems.append(
            list(
                range(
                    start_index,
                    min(len(groups), start_index + events_per_system),
                )
            )
        )

    system_total_units = [
        sum(group_units[index] for index in system) for system in systems
    ]

    max_total_units = max(system_total_units)
    system_count = len(systems)

    figure_width = max(13.0, min(60.0, 3.0 + max_total_units * 0.45))
    figure_height = max(5.0, system_count * 3.8)  # More height for noteheads

    fig, axes = plt.subplots(
        system_count,
        1,
        figsize=(figure_width, figure_height),
        squeeze=False,
    )

    axes_list = [axes[index][0] for index in range(system_count)]
    x_start = 1.0
    string_label_x = x_start - 0.6

    for system_index, ax in enumerate(axes_list):
        group_indexes = systems[system_index]
        line_end_x = x_start + max_total_units + 0.4

        # Draw vertical grid lines
        for unit_index in range(max_total_units + 1):
            grid_x = x_start + unit_index
            is_group_of_eight = unit_index % 8 == 0

            ax.axvline(
                grid_x,
                color="#C9D4DF" if is_group_of_eight else "#E8EDF2",
                linewidth=0.85 if is_group_of_eight else 0.55,
                zorder=0,
            )

        # Draw strings with labels
        for string_index, (string_name, open_pitch) in enumerate(GUITAR_STRINGS):
            ax.plot(
                [x_start - 0.25, line_end_x],
                [string_index, string_index],
                color="#222222",
                linewidth=0.9,
                zorder=1,
            )

            ax.text(
                string_label_x,
                string_index,
                string_name,
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=3,
            )

            ax.text(
                string_label_x - 0.55,
                string_index,
                pretty_midi.note_number_to_name(open_pitch),
                fontsize=7,
                color="#888888",
                ha="center",
                va="center",
                zorder=3,
            )

        cursor_units = 0

        # Draw notes with noteheads
        for local_index, group_index in enumerate(group_indexes, start=1):
            group = groups[group_index]
            units = group_units[group_index]
            x = x_start + cursor_units + 0.5
            cursor_units += units

            notes_for_draw = sorted(
                group,
                key=lambda note: note.candidate_rank,
                reverse=True,
            )

            drawn_rows: set[int] = set()

            for note in notes_for_draw:
                position = choose_guitar_position(note.pitch)

                if position is None:
                    continue

                visual_row = position.string_index
                note_x = x

                if note.candidate_rank == 2 and visual_row in drawn_rows:
                    note_x = x - 0.30

                note_units = duration_units_from_seconds(
                    duration_seconds=note.duration_seconds,
                    reference_seconds=reference_seconds,
                    use_snap=use_snap,
                )

                # Draw notehead above the string
                notehead_y = visual_row + 0.55
                draw_notehead(
                    ax=ax,
                    x=note_x,
                    y=notehead_y,
                    units=note_units,
                    is_rank_1=(note.candidate_rank == 1),
                )

                # Draw fret number on the string
                if note.candidate_rank == 1:
                    text = str(position.fret)
                    text_color = "#173F6F"
                    edge_color = "#0D2745"
                    font_size = 10
                    font_weight = "bold"
                    zorder = 6
                else:
                    text = f"({position.fret})"
                    text_color = "#684B9B"
                    edge_color = "#B9A5E8"
                    font_size = 8
                    font_weight = "normal"
                    zorder = 5

                ax.text(
                    note_x,
                    visual_row,
                    text,
                    fontsize=font_size,
                    color=text_color,
                    fontweight=font_weight,
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.10",
                        "facecolor": "white",
                        "edgecolor": edge_color,
                        "linewidth": 0.6,
                        "alpha": 0.92,
                    },
                    zorder=zorder,
                )

                # Draw duration line if note is held
                if note_units >= 2:
                    dash_start_x = note_x + 0.30
                    dash_end_x = x + note_units - 0.35

                    if dash_end_x > dash_start_x:
                        ax.plot(
                            [dash_start_x, dash_end_x],
                            [visual_row, visual_row],
                            color=(
                                "#173F6F"
                                if note.candidate_rank == 1
                                else "#B9A5E8"
                            ),
                            linewidth=(2.0 if note.candidate_rank == 1 else 1.4),
                            solid_capstyle="butt",
                            zorder=zorder - 1,
                        )

                drawn_rows.add(visual_row)

            # Draw unit count below
            duration_note = pick_group_duration_note(group)

            ax.text(
                x,
                -0.62,
                str(units),
                fontsize=8,
                color="#555555",
                ha="center",
                va="top",
            )

            if (
                local_index == 1
                or local_index % 4 == 0
                or group_index == len(groups) - 1
            ):
                ax.text(
                    x,
                    -1.12,
                    f"{duration_note.start_seconds:.2f}s",
                    fontsize=7,
                    color="#888888",
                    ha="center",
                    va="top",
                )

        ax.set_xlim(string_label_x - 1.1, x_start + max_total_units + 0.8)
        ax.set_ylim(-1.75, 6.8)  # Extended upper limit for noteheads
        ax.set_xticks([])
        ax.set_yticks([])

        system_units = system_total_units[system_index]

        ax.set_title(
            f"System {system_index + 1} | "
            f"{len(group_indexes)} onsets | "
            f"{system_units} eighth-note units",
            loc="left",
            fontsize=9,
            color="#444444",
            pad=6,
        )

        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Guitar Tab with Standard Noteheads | Standard Tuning, low E at bottom",
        fontsize=15,
        y=0.995,
    )

    fig.text(
        0.5,
        0.012,
        (
            f"One unit = one eighth note = {reference_seconds:.3f}s ({reference_mode}). "
            "Noteheads show duration: filled oval + stem = eighth/quarter, "
            "hollow oval + stem = half, hollow oval alone = whole. "
            "Dot after notehead = dotted note. "
            "Rank 1 (blue/dark), Rank 2 (purple, in parentheses)."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout(rect=(0, 0.05, 1, 0.965))

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def build_output_paths(
    output_root: Path,
    source_path: Path,
    max_candidates_per_onset: int,
) -> dict[str, dict[str, Path]]:
    source_name = source_path.stem
    mode_label = f"top{max_candidates_per_onset}"

    source_dir = output_root / "by_source" / source_name
    source_dir.mkdir(parents=True, exist_ok=True)

    aggregate_dirs = {
        key: output_root / folder_name for key, folder_name in OUTPUT_FOLDERS.items()
    }

    for directory in aggregate_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    file_names = {
        "basic_pitch_original_midi": f"{source_name}_01_basic_pitch_original.mid",
        "raw_candidates_midi": f"{source_name}_02_raw_{mode_label}_candidates.mid",
        "raw_notes_csv": f"{source_name}_03_raw_{mode_label}_notes.csv",
        "raw_pianoroll_png": f"{source_name}_04_raw_{mode_label}_pianoroll.png",
        "raw_pianoroll_event_grid_csv": f"{source_name}_05_raw_{mode_label}_event_grid.csv",
        "raw_guitar_tab_preview_png": f"{source_name}_06_raw_{mode_label}_guitar_tab.png",
        "raw_guitar_tab_csv": f"{source_name}_07_raw_{mode_label}_guitar_tab.csv",
        "raw_guitar_tab_with_noteheads_png": f"{source_name}_08_raw_{mode_label}_guitar_tab_noteheads.png", 
    }

    return {
        "source": {
            key: source_dir / file_name for key, file_name in file_names.items()
        },
        "aggregate": {
            key: aggregate_dirs[key] / file_name for key, file_name in file_names.items()
        },
    }


def copy_outputs_to_aggregate(
    source_paths: dict[str, Path],
    aggregate_paths: dict[str, Path],
) -> None:
    for key, source_file in source_paths.items():
        shutil.copy2(source_file, aggregate_paths[key])


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
    merge_same_pitch: bool,
    merge_max_gap_ms: float,
    merge_velocity_jump: float,
    png_note_width_scale: float,
    png_min_block_ms: float,
    duration_reference: str,
    duration_percentile: float,
    snap_units: bool,
    tab_events_per_system: int,
) -> int:
    # try:
    #     import librosa
    #     test_audio, _ = librosa.load(str(source_path), sr=22050, mono=True, duration=1.0)
    #     if len(test_audio) == 0:
    #         raise ValueError("Audio file is empty or unreadable")
    # except Exception as e:
    #     raise ValueError(f"Cannot read audio file: {e}")

    basic_pitch_midi, note_events = get_basic_pitch_result(
        source_path=source_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length_ms=min_duration_ms,
    )

    print("  [STEP 2/7] Reading confidence-ranked candidates")

    candidate_notes = collect_candidate_notes(
        midi_data=basic_pitch_midi,
        note_events=note_events,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        min_duration_seconds=min_duration_ms / 1000.0,
    )

    if not candidate_notes:
        raise ValueError("No usable note candidates were detected.")

    print(
        f"  [INFO] Keeping top {max_candidates_per_onset} candidate(s) per onset."
    )

    raw_notes = make_raw_top_candidates(
        candidates=candidate_notes,
        max_candidates_per_onset=max_candidates_per_onset,
        same_start_tolerance_seconds=same_start_tolerance_ms / 1000.0,
    )

    if not raw_notes:
        raise ValueError("No notes remain after candidate filtering.")

    if merge_same_pitch:
        segment_count = len(raw_notes)

        raw_notes = merge_sustained_same_pitch_notes(
            raw_notes=raw_notes,
            max_gap_seconds=merge_max_gap_ms / 1000.0,
            velocity_jump_threshold=merge_velocity_jump,
        )

        print(
            f"  [INFO] Sustained-note merge: {segment_count} segments -> {len(raw_notes)} notes"
        )

    reference_seconds = calculate_duration_reference(
        raw_notes=raw_notes,
        reference_mode=duration_reference,
        percentile=duration_percentile,
    )

    # 调试：输出实际的时长分布
    durations = sorted([note.duration_seconds for note in raw_notes])
    print(f"\n  [DEBUG] Duration statistics:")
    print(f"    Shortest: {durations[0]:.4f}s")
    print(f"    10th percentile: {durations[len(durations) // 10]:.4f}s")
    print(f"    Median: {durations[len(durations) // 2]:.4f}s")
    print(f"    Reference chosen: {reference_seconds:.4f}s")
    print(f"    After *2.0: {reference_seconds * 2.0:.4f}s")

    # 看看实际计算出的 units
    sample_units = [
        duration_units_from_seconds(d, reference_seconds, False)
        for d in durations[:10]
    ]
    print(f"    First 10 notes' units (no snap): {sample_units}\n")

    print(
        f"  [INFO] Eighth-note reference: {reference_seconds:.3f}s ({duration_reference})"
    )

    output_paths = build_output_paths(
        output_root=output_root,
        source_path=source_path,
        max_candidates_per_onset=max_candidates_per_onset,
    )

    print("  [STEP 3/7] Writing MIDI files")

    basic_pitch_midi.write(
        str(output_paths["source"]["basic_pitch_original_midi"])
    )

    write_raw_candidate_midi(
        output_path=output_paths["source"]["raw_candidates_midi"],
        notes=raw_notes,
    )

    print("  [STEP 4/7] Writing raw note CSV files")

    write_raw_notes_csv(
        output_path=output_paths["source"]["raw_notes_csv"],
        raw_notes=raw_notes,
    )

    write_raw_pianoroll_event_grid_csv(
        output_path=output_paths["source"]["raw_pianoroll_event_grid_csv"],
        raw_notes=raw_notes,
    )

    print("  [STEP 5/7] Rendering piano-roll PNG")

    render_raw_piano_roll(
        output_path=output_paths["source"]["raw_pianoroll_png"],
        raw_notes=raw_notes,
        note_width_scale=png_note_width_scale,
        minimum_visual_width_seconds=png_min_block_ms / 1000.0,
    )

    print("  [STEP 6/7] Writing guitar-tab CSV")

    write_guitar_tab_csv(
        output_path=output_paths["source"]["raw_guitar_tab_csv"],
        raw_notes=raw_notes,
        same_start_tolerance_seconds=same_start_tolerance_ms / 1000.0,
        reference_seconds=reference_seconds,
        reference_mode=duration_reference,
        use_snap=snap_units,
    )
  
    print("  [STEP 7/8] Rendering guitar-tab PNG")
    
    render_guitar_tab_preview(
        output_path=output_paths["source"]["raw_guitar_tab_preview_png"],
        raw_notes=raw_notes,
        same_start_tolerance_seconds=same_start_tolerance_ms / 1000.0,
        reference_seconds=reference_seconds,
        reference_mode=duration_reference,
        use_snap=snap_units,
        events_per_system=tab_events_per_system,
    )
    
    print("  [STEP 8/8] Rendering guitar-tab with noteheads PNG")
    
    render_guitar_tab_with_noteheads(
        output_path=output_paths["source"]["raw_guitar_tab_with_noteheads_png"],
        raw_notes=raw_notes,
        same_start_tolerance_seconds=same_start_tolerance_ms / 1000.0,
        reference_seconds=reference_seconds,
        reference_mode=duration_reference,
        use_snap=snap_units,
        events_per_system=tab_events_per_system,
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
            f"[ERROR] Input directory does not exist: {args.input_dir.resolve()}"
        )
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mp3_files = collect_mp3_files(args.input_dir)

    if not mp3_files:
        print(f"[WARN] No MP3 files found in: {args.input_dir.resolve()}")
        return

    print("=" * 72)
    print("[INFO] MP3 -> Raw Top Confidence Candidates + Guitar Tab Draft")
    print("[INFO] No BPM detection.")
    print("[INFO] No key signature and no transposition.")
    print(f"[INFO] Input : {args.input_dir.resolve()}")
    print(f"[INFO] Output: {args.output_dir.resolve()}")
    print(f"[INFO] Candidates per onset: {args.max_candidates_per_onset}")
    print(f"[INFO] Same-start tolerance: {args.same_start_tolerance_ms:g} ms")
    print(
        f"[INFO] Minimum detected note duration: {args.min_duration_ms:g} ms"
    )
    print(f"[INFO] Pitch range: MIDI {args.pitch_min}-{args.pitch_max}")
    print(
        f"[INFO] Sustained-note merge: {'on' if args.merge_same_pitch else 'off'}"
    )

    if args.merge_same_pitch:
        print(f"[INFO] Merge max gap: {args.merge_max_gap_ms:g} ms")
        print(
            f"[INFO] Merge velocity-jump limit: {args.merge_velocity_jump:g}"
        )

    print(
        f"[INFO] Piano-roll visual width: x{args.png_note_width_scale:g}, "
        f"minimum {args.png_min_block_ms:g} ms"
    )
    print(f"[INFO] Duration reference: {args.duration_reference}")

    if args.duration_reference == "percentile":
        print(f"[INFO] Duration percentile: {args.duration_percentile:g}")

    print(
        f"[INFO] Duration unit snapping: {'on' if args.snap_duration_units else 'off'}"
    )
    print(
        f"[INFO] Guitar tuning: standard, low E bottom to high e top, "
        f"{GUITAR_MAX_FRET} frets"
    )
    print(f"[INFO] Tab onsets per system: {args.tab_events_per_system}")
    print("=" * 72)

    success_count = 0

    for index, source_path in enumerate(mp3_files, start=1):
        print()
        print(f"[{index}/{len(mp3_files)}] Processing: {source_path.name}")

        try:
            note_count = process_one_file(
                source_path=source_path,
                output_root=args.output_dir,
                max_candidates_per_onset=args.max_candidates_per_onset,
                same_start_tolerance_ms=args.same_start_tolerance_ms,
                min_duration_ms=args.min_duration_ms,
                onset_threshold=args.onset_threshold,
                frame_threshold=args.frame_threshold,
                pitch_min=args.pitch_min,
                pitch_max=args.pitch_max,
                merge_same_pitch=args.merge_same_pitch,
                merge_max_gap_ms=args.merge_max_gap_ms,
                merge_velocity_jump=args.merge_velocity_jump,
                png_note_width_scale=args.png_note_width_scale,
                png_min_block_ms=args.png_min_block_ms,
                duration_reference=args.duration_reference,
                duration_percentile=args.duration_percentile,
                snap_units=args.snap_duration_units,
                tab_events_per_system=args.tab_events_per_system,
            )

            print(f"  [OK] Done. Final notes: {note_count}")
            success_count += 1

        except Exception as error:
            print(f"  [FAIL] {error}", file=sys.stderr)

    print()
    print("=" * 72)
    print(f"[DONE] Successfully processed: {success_count}/{len(mp3_files)}")
    print("=" * 72)

if __name__ == "__main__":
    main()
