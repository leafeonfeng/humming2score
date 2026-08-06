from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Server / terminal environment: render PNG without opening a GUI window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

import pretty_midi
from basic_pitch.inference import predict


STEM_FILE_NAME = "vocals.wav"

CSV_FIELD_ORDER = [
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
    """A Basic Pitch note event before Top-1 / Top-2 filtering."""

    start_seconds: float
    end_seconds: float
    pitch: int
    velocity: int
    confidence: float


@dataclass
class FinalNote:
    """A final retained vocal note.

    candidate_rank:
        1 = highest confidence candidate in an onset group.
        2 = second-highest confidence candidate in an onset group.
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
            "Step 2: Transcribe Demucs-separated vocals.wav files with "
            "Basic Pitch. Output raw-time Top-1 / Top-2 candidate MIDI, "
            "CSV and piano-roll PNG files."
        )
    )

    parser.add_argument(
        "--jam-root",
        type=Path,
        default=Path("output_jam/by_source"),
        help=(
            "Directory containing one folder per Jam recording. "
            "Default: output_jam/by_source"
        ),
    )

    parser.add_argument(
        "--source-name",
        type=str,
        default=None,
        help=(
            "Only process one Jam folder name, for example: jam_001. "
            "Default: process all folders under --jam-root."
        ),
    )

    parser.add_argument(
        "--max-candidates-per-onset",
        type=int,
        choices=[1, 2],
        default=2,
        help=(
            "Keep 1 or 2 confidence-ranked notes for each onset group. "
            "Default: 2"
        ),
    )

    parser.add_argument(
        "--same-start-tolerance-ms",
        type=float,
        default=25.0,
        help=(
            "Notes starting within this number of milliseconds are treated "
            "as one onset group. Default: 25"
        ),
    )

    parser.add_argument(
        "--min-duration-ms",
        type=float,
        default=70.0,
        help=(
            "Discard detected notes shorter than this duration. "
            "Default: 70"
        ),
    )

    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.55,
        help=(
            "Basic Pitch onset threshold. Higher values reduce false notes "
            "but can miss quiet notes. Default: 0.55"
        ),
    )

    parser.add_argument(
        "--frame-threshold",
        type=float,
        default=0.35,
        help=(
            "Basic Pitch frame threshold. Higher values reduce sustained "
            "noise but can shorten notes. Default: 0.35"
        ),
    )

    parser.add_argument(
        "--pitch-min",
        type=int,
        default=36,
        help=(
            "Minimum vocal MIDI pitch to retain. "
            "Default: 36 = C2"
        ),
    )

    parser.add_argument(
        "--pitch-max",
        type=int,
        default=84,
        help=(
            "Maximum vocal MIDI pitch to retain. "
            "Default: 84 = C6"
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

    return args


def format_seconds(seconds: float) -> str:
    """Format seconds as MM:SS."""
    total_seconds = max(0, int(seconds))

    minutes = total_seconds // 60
    seconds_remaining = total_seconds % 60

    return f"{minutes:02d}:{seconds_remaining:02d}"


def collect_source_directories(
    jam_root: Path,
    source_name: str | None,
) -> list[Path]:
    """Find Jam source folders containing stems/vocals.wav."""
    if source_name is not None:
        source_dir = jam_root / source_name

        if not source_dir.exists():
            raise FileNotFoundError(
                "Requested --source-name folder does not exist:\n"
                f"{source_dir}"
            )

        return [source_dir]

    return sorted(
        path
        for path in jam_root.iterdir()
        if path.is_dir()
    )


def get_vocals_path(source_dir: Path) -> Path:
    """Return the expected vocals.wav path for a Jam source folder."""
    return source_dir / "stems" / STEM_FILE_NAME


def run_basic_pitch(
    vocals_path: Path,
    onset_threshold: float,
    frame_threshold: float,
    min_duration_ms: float,
    pitch_min: int,
    pitch_max: int,
) -> tuple[pretty_midi.PrettyMIDI, list]:
    """Run Basic Pitch on separated vocals.

    Important:
    - This script does not convert audio time to BPM.
    - This script does not quantize rhythm.
    - Basic Pitch's extracted note event amplitude is used as a relative
      confidence value to rank simultaneous pitch candidates.
    """
    minimum_frequency = pretty_midi.note_number_to_hz(pitch_min)
    maximum_frequency = pretty_midi.note_number_to_hz(pitch_max)

    print(
        "  [STEP 1/4] Basic Pitch vocal transcription started.",
        flush=True,
    )

    print(
        "  [INFO] Vocal pitch range: "
        f"{pretty_midi.note_number_to_name(pitch_min)} "
        f"(MIDI {pitch_min}) -> "
        f"{pretty_midi.note_number_to_name(pitch_max)} "
        f"(MIDI {pitch_max})",
        flush=True,
    )

    print(
        "  [INFO] Basic Pitch may take time on the first file while "
        "loading its model.",
        flush=True,
    )

    _, midi_data, note_events = predict(
        str(vocals_path),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_duration_ms,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        multiple_pitch_bends=False,
        melodia_trick=True,
    )

    return midi_data, note_events


def collect_midi_notes(
    midi_data: pretty_midi.PrettyMIDI,
) -> list[pretty_midi.Note]:
    """Collect non-drum notes from Basic Pitch generated MIDI."""
    notes: list[pretty_midi.Note] = []

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue

        notes.extend(instrument.notes)

    return sorted(
        notes,
        key=lambda note: (
            note.start,
            note.end,
            note.pitch,
        ),
    )


def confidence_to_velocity(confidence: float) -> int:
    """Use confidence as fallback velocity if MIDI matching fails."""
    confidence = max(0.0, min(1.0, confidence))

    velocity = round(1 + confidence * 126)

    return max(1, min(127, velocity))


def find_matching_velocity(
    midi_notes: list[pretty_midi.Note],
    start_seconds: float,
    end_seconds: float,
    pitch: int,
) -> int:
    """Find the nearest Basic Pitch MIDI note with the same pitch."""
    matching_notes = [
        note
        for note in midi_notes
        if note.pitch == pitch
    ]

    if not matching_notes:
        return 0

    closest_note = min(
        matching_notes,
        key=lambda note: (
            abs(note.start - start_seconds)
            + abs(note.end - end_seconds)
        ),
    )

    timing_difference = (
        abs(closest_note.start - start_seconds)
        + abs(closest_note.end - end_seconds)
    )

    # Basic Pitch MIDI event timing may differ slightly from note_events.
    if timing_difference <= 0.06:
        return closest_note.velocity

    return 0


def collect_candidate_notes(
    midi_data: pretty_midi.PrettyMIDI,
    note_events: list,
    pitch_min: int,
    pitch_max: int,
    minimum_duration_seconds: float,
) -> list[CandidateNote]:
    """Read candidate note events produced by Basic Pitch."""
    midi_notes = collect_midi_notes(midi_data)

    candidates: list[CandidateNote] = []

    for event in note_events:
        # Expected Basic Pitch event form:
        # (start_seconds, end_seconds, midi_pitch, amplitude, pitch_bends)
        if len(event) < 4:
            continue

        start_seconds = float(event[0])
        end_seconds = float(event[1])
        pitch = int(event[2])
        confidence = float(event[3])

        duration_seconds = end_seconds - start_seconds

        if duration_seconds < minimum_duration_seconds:
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
        key=lambda note: (
            note.start_seconds,
            -note.confidence,
            note.pitch,
        ),
    )


def group_notes_by_onset(
    candidates: list[CandidateNote],
    tolerance_seconds: float,
) -> list[list[CandidateNote]]:
    """Group notes with approximately identical start times."""
    if not candidates:
        return []

    sorted_candidates = sorted(
        candidates,
        key=lambda note: (
            note.start_seconds,
            -note.confidence,
            note.pitch,
        ),
    )

    groups: list[list[CandidateNote]] = []

    current_group: list[CandidateNote] = [
        sorted_candidates[0]
    ]

    group_anchor_time = sorted_candidates[0].start_seconds

    for candidate in sorted_candidates[1:]:
        if (
            candidate.start_seconds
            - group_anchor_time
            <= tolerance_seconds
        ):
            current_group.append(candidate)

        else:
            groups.append(current_group)

            current_group = [candidate]

            group_anchor_time = candidate.start_seconds

    groups.append(current_group)

    return groups


def deduplicate_pitches_in_group(
    group: list[CandidateNote],
) -> list[CandidateNote]:
    """Keep only the highest-confidence event for each MIDI pitch."""
    best_by_pitch: dict[int, CandidateNote] = {}

    for candidate in group:
        previous = best_by_pitch.get(candidate.pitch)

        if previous is None:
            best_by_pitch[candidate.pitch] = candidate
            continue

        if candidate.confidence > previous.confidence:
            best_by_pitch[candidate.pitch] = candidate

    return list(best_by_pitch.values())


def make_top_candidates(
    candidates: list[CandidateNote],
    max_candidates_per_onset: int,
    same_start_tolerance_seconds: float,
) -> list[FinalNote]:
    """Keep Top-1 or Top-2 notes in every raw onset group.

    No tempo conversion.
    No beat grid.
    No rhythmic quantization.

    If a note overlaps the next detected onset, its end is shortened to the
    next onset time. This makes the piano-roll events easier to inspect.
    """
    if not candidates:
        return []

    onset_groups = group_notes_by_onset(
        candidates=candidates,
        tolerance_seconds=same_start_tolerance_seconds,
    )

    final_notes: list[FinalNote] = []

    for group_index, group in enumerate(onset_groups):
        unique_pitch_group = deduplicate_pitches_in_group(group)

        ranked_group = sorted(
            unique_pitch_group,
            key=lambda note: (
                note.confidence,
                note.velocity,
                note.end_seconds - note.start_seconds,
                note.pitch,
            ),
            reverse=True,
        )

        selected_notes = ranked_group[
            :max_candidates_per_onset
        ]

        next_onset_time: float | None = None

        if group_index + 1 < len(onset_groups):
            next_onset_time = onset_groups[
                group_index + 1
            ][0].start_seconds

        for rank, candidate in enumerate(
            selected_notes,
            start=1,
        ):
            final_end_seconds = candidate.end_seconds

            # Prevent one visual note rectangle from extending across
            # the next detected note onset.
            if (
                next_onset_time is not None
                and final_end_seconds > next_onset_time
            ):
                final_end_seconds = next_onset_time

            if final_end_seconds <= candidate.start_seconds:
                continue

            final_notes.append(
                FinalNote(
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
                    candidate_rank=rank,
                )
            )

    return sorted(
        final_notes,
        key=lambda note: (
            note.start_seconds,
            note.candidate_rank,
            note.pitch,
        ),
    )


def write_filtered_midi(
    output_path: Path,
    notes: list[FinalNote],
) -> None:
    """Write Top-1 / Top-2 candidate notes as a MIDI file."""
    midi_data = pretty_midi.PrettyMIDI(
        initial_tempo=120.0
    )

    instrument = pretty_midi.Instrument(
        program=52,
        is_drum=False,
        name="Vocal Raw Candidates",
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


def write_notes_csv(
    output_path: Path,
    notes: list[FinalNote],
) -> None:
    """Write one retained note per CSV row."""
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELD_ORDER,
        )

        writer.writeheader()

        for note_index, note in enumerate(notes, start=1):
            writer.writerow(
                {
                    "note_index": note_index,
                    "note_name": note.note_name,
                    "midi_pitch": note.pitch,
                    "velocity": note.velocity,
                    "confidence": (
                        f"{note.confidence:.6f}"
                    ),
                    "candidate_rank": note.candidate_rank,
                    "start_seconds": (
                        f"{note.start_seconds:.6f}"
                    ),
                    "end_seconds": (
                        f"{note.end_seconds:.6f}"
                    ),
                    "duration_seconds": (
                        f"{note.duration_seconds:.6f}"
                    ),
                }
            )


def build_event_times(
    notes: list[FinalNote],
) -> list[float]:
    """Build a raw event-time axis from note starts and ends."""
    event_times = {
        round(note.start_seconds, 6)
        for note in notes
    }

    event_times.update(
        round(note.end_seconds, 6)
        for note in notes
    )

    return sorted(event_times)


def write_event_grid_csv(
    output_path: Path,
    notes: list[FinalNote],
) -> None:
    """Write an event-based piano-roll CSV.

    Columns are actual detected start/end times in seconds.
    This is deliberately not BPM-quantized.
    """
    if not notes:
        raise ValueError("No notes available for event-grid CSV.")

    minimum_pitch = max(
        0,
        min(note.pitch for note in notes) - 2,
    )

    maximum_pitch = min(
        127,
        max(note.pitch for note in notes) + 2,
    )

    event_times = build_event_times(notes)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)

        time_headers = [
            f"{event_time:.6f}s"
            for event_time in event_times
        ]

        writer.writerow(
            [
                "pitch_or_time",
                *time_headers,
            ]
        )

        for pitch in range(
            maximum_pitch,
            minimum_pitch - 1,
            -1,
        ):
            row: list[str] = [
                f"{pretty_midi.note_number_to_name(pitch)} "
                f"(MIDI {pitch})"
            ]

            for event_time in event_times:
                active_notes = [
                    note
                    for note in notes
                    if note.pitch == pitch
                    and note.start_seconds <= event_time
                    and event_time < note.end_seconds
                ]

                if not active_notes:
                    row.append("")
                    continue

                # If overlapping same-pitch candidates occur,
                # retain the largest velocity in the grid cell.
                row.append(
                    str(
                        max(
                            note.velocity
                            for note in active_notes
                        )
                    )
                )

            writer.writerow(row)


def render_piano_roll(
    output_path: Path,
    notes: list[FinalNote],
) -> None:
    """Render raw piano-roll PNG.

    Rank 1:
        Blue; darker blue means higher velocity.

    Rank 2:
        Fixed light purple; velocity is intentionally not represented.
    """
    if not notes:
        raise ValueError("No notes available for piano-roll PNG.")

    minimum_pitch = max(
        0,
        min(note.pitch for note in notes) - 2,
    )

    maximum_pitch = min(
        127,
        max(note.pitch for note in notes) + 2,
    )

    total_duration = max(
        note.end_seconds
        for note in notes
    )

    figure_width = max(
        11,
        min(30, 9 + total_duration * 0.32),
    )

    figure_height = max(
        5,
        min(
            14,
            3.5 + (maximum_pitch - minimum_pitch) * 0.28,
        ),
    )

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height),
    )

    velocity_normalizer = colors.Normalize(
        vmin=1,
        vmax=127,
    )

    blue_colormap = plt.cm.Blues

    rank_2_fill_color = "#C4B2E8"
    rank_2_border_color = "#74559D"

    # Draw Rank 2 first. Rank 1 is then visible above it if overlapping.
    notes_to_draw = sorted(
        notes,
        key=lambda note: (
            note.candidate_rank,
            note.start_seconds,
            note.pitch,
        ),
        reverse=True,
    )

    for note in notes_to_draw:
        if note.candidate_rank == 1:
            face_color = blue_colormap(
                velocity_normalizer(note.velocity)
            )

            edge_color = "#0D3B66"
            alpha = 0.96

        else:
            face_color = rank_2_fill_color
            edge_color = rank_2_border_color
            alpha = 0.86

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

        axis.add_patch(rectangle)

    axis.set_xlim(
        -0.05,
        total_duration + 0.15,
    )

    axis.set_ylim(
        minimum_pitch - 0.6,
        maximum_pitch + 0.6,
    )

    axis.set_xlabel("Original audio time (seconds)")
    axis.set_ylabel("Vocal pitch")

    axis.set_title(
        "Vocals Raw Piano Roll | "
        "Rank 1: Blue by Velocity | "
        "Rank 2: Light Purple"
    )

    # Horizontal lines are pitch guides.
    for pitch in range(
        minimum_pitch,
        maximum_pitch + 1,
    ):
        axis.axhline(
            pitch,
            color="#E0E0E0",
            linewidth=0.45,
            zorder=0,
        )

    # Vertical lines are one-second visual guides only.
    # They do not alter note timing or quantize the data.
    for second in range(
        int(math.ceil(total_duration)) + 1
    ):
        axis.axvline(
            second,
            color="#D7E0E8",
            linewidth=0.65,
            zorder=0,
        )

    pitch_ticks = list(
        range(
            minimum_pitch,
            maximum_pitch + 1,
        )
    )

    axis.set_yticks(pitch_ticks)

    axis.set_yticklabels(
        [
            pretty_midi.note_number_to_name(pitch)
            for pitch in pitch_ticks
        ]
    )

    velocity_mapper = ScalarMappable(
        norm=velocity_normalizer,
        cmap=blue_colormap,
    )

    velocity_mapper.set_array([])

    colorbar = figure.colorbar(
        velocity_mapper,
        ax=axis,
        pad=0.015,
    )

    colorbar.set_label(
        "Rank 1 MIDI velocity "
        "(light blue -> dark blue)",
        rotation=270,
        labelpad=18,
    )

    rank_1_low_legend = Rectangle(
        (0, 0),
        1,
        1,
        facecolor=blue_colormap(
            velocity_normalizer(25)
        ),
        edgecolor="#0D3B66",
        label="Rank 1: low velocity",
    )

    rank_1_high_legend = Rectangle(
        (0, 0),
        1,
        1,
        facecolor=blue_colormap(
            velocity_normalizer(110)
        ),
        edgecolor="#0D3B66",
        label="Rank 1: high velocity",
    )

    rank_2_legend = Rectangle(
        (0, 0),
        1,
        1,
        facecolor=rank_2_fill_color,
        edgecolor=rank_2_border_color,
        label="Rank 2: second confidence candidate",
    )

    axis.legend(
        handles=[
            rank_1_low_legend,
            rank_1_high_legend,
            rank_2_legend,
        ],
        loc="upper right",
        framealpha=0.95,
    )

    axis.grid(False)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def write_readme(
    output_path: Path,
    source_name: str,
    vocals_path: Path,
    notes: list[FinalNote],
    args: argparse.Namespace,
) -> None:
    """Write output and parameter explanation."""
    rank_1_count = sum(
        note.candidate_rank == 1
        for note in notes
    )

    rank_2_count = sum(
        note.candidate_rank == 2
        for note in notes
    )

    text = f"""Jam Step 2: Vocal Transcription Result
======================================

Source name:
{source_name}

Input vocal stem:
{vocals_path}

Processing rules:
- Source timing is retained in seconds.
- No BPM conversion.
- No beat-grid quantization.
- No rhythmic correction.
- Notes were grouped when their start times differ by no more than:
  {args.same_start_tolerance_ms:g} ms
- Up to {args.max_candidates_per_onset} confidence-ranked candidate(s)
  were retained per onset group.

Visual rules in PNG:
- Rank 1: Blue. Darker blue means higher MIDI velocity.
- Rank 2: Fixed light purple. It does not express velocity.

Basic Pitch settings:
- onset_threshold: {args.onset_threshold}
- frame_threshold: {args.frame_threshold}
- min_duration_ms: {args.min_duration_ms}
- vocal pitch range:
  {pretty_midi.note_number_to_name(args.pitch_min)} (MIDI {args.pitch_min})
  to
  {pretty_midi.note_number_to_name(args.pitch_max)} (MIDI {args.pitch_max})

Result:
- Total retained notes: {len(notes)}
- Rank 1 notes: {rank_1_count}
- Rank 2 notes: {rank_2_count}

Files:
- 01_basic_pitch_original.mid
  Original Basic Pitch MIDI before Top-1 / Top-2 filtering.

- 02_vocals_top2_candidates.mid
  Filtered MIDI. MIDI stores pitch, timing and velocity;
  it does not natively store confidence or candidate rank.

- 03_vocals_top2_notes.csv
  One note per row, including confidence and candidate_rank.

- 04_vocals_top2_pianoroll.png
  Visual inspection chart.

- 05_vocals_top2_event_grid.csv
  Event-based piano-roll grid using actual note start/end times.
"""

    output_path.write_text(
        text,
        encoding="utf-8",
    )


def process_one_source(
    source_dir: Path,
    args: argparse.Namespace,
) -> int:
    """Transcribe one Jam source folder's vocals.wav."""
    source_name = source_dir.name

    vocals_path = get_vocals_path(source_dir)

    if not vocals_path.exists():
        raise FileNotFoundError(
            "Could not find vocal stem:\n"
            f"{vocals_path}\n"
            "\n"
            "Run Step 1 stem separation first."
        )

    output_dir = (
        source_dir
        / "transcription"
        / "vocals"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    basic_pitch_midi_path = (
        output_dir
        / "01_basic_pitch_original.mid"
    )

    filtered_midi_path = (
        output_dir
        / "02_vocals_top2_candidates.mid"
    )

    notes_csv_path = (
        output_dir
        / "03_vocals_top2_notes.csv"
    )

    piano_roll_png_path = (
        output_dir
        / "04_vocals_top2_pianoroll.png"
    )

    event_grid_csv_path = (
        output_dir
        / "05_vocals_top2_event_grid.csv"
    )

    readme_path = output_dir / "README.txt"

    midi_data, note_events = run_basic_pitch(
        vocals_path=vocals_path,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        min_duration_ms=args.min_duration_ms,
        pitch_min=args.pitch_min,
        pitch_max=args.pitch_max,
    )

    print(
        "  [STEP 2/4] Reading and ranking vocal note candidates.",
        flush=True,
    )

    candidates = collect_candidate_notes(
        midi_data=midi_data,
        note_events=note_events,
        pitch_min=args.pitch_min,
        pitch_max=args.pitch_max,
        minimum_duration_seconds=(
            args.min_duration_ms / 1000.0
        ),
    )

    if not candidates:
        raise ValueError(
            "No usable vocal candidates were detected.\n"
            "Try lowering --onset-threshold or --frame-threshold, "
            "or widening --pitch-min / --pitch-max."
        )

    final_notes = make_top_candidates(
        candidates=candidates,
        max_candidates_per_onset=(
            args.max_candidates_per_onset
        ),
        same_start_tolerance_seconds=(
            args.same_start_tolerance_ms / 1000.0
        ),
    )

    if not final_notes:
        raise ValueError(
            "No notes remain after Top-1 / Top-2 filtering."
        )

    print(
        "  [STEP 3/4] Writing MIDI and CSV output.",
        flush=True,
    )

    # Save Basic Pitch output before custom Top-1 / Top-2 filtering.
    midi_data.write(str(basic_pitch_midi_path))

    write_filtered_midi(
        output_path=filtered_midi_path,
        notes=final_notes,
    )

    write_notes_csv(
        output_path=notes_csv_path,
        notes=final_notes,
    )

    write_event_grid_csv(
        output_path=event_grid_csv_path,
        notes=final_notes,
    )

    print(
        "  [STEP 4/4] Rendering vocal piano-roll PNG.",
        flush=True,
    )

    render_piano_roll(
        output_path=piano_roll_png_path,
        notes=final_notes,
    )

    write_readme(
        output_path=readme_path,
        source_name=source_name,
        vocals_path=vocals_path,
        notes=final_notes,
        args=args,
    )

    return len(final_notes)


def main() -> None:
    args = parse_args()

    if not args.jam_root.exists():
        print(
            "[ERROR] Jam source root does not exist:\n"
            f"  {args.jam_root.resolve()}\n"
            "\n"
            "Run Step 1 first, or provide --jam-root.",
            flush=True,
        )

        sys.exit(1)

    try:
        source_directories = collect_source_directories(
            jam_root=args.jam_root,
            source_name=args.source_name,
        )

    except FileNotFoundError as error:
        print(f"[ERROR]\n{error}", flush=True)
        sys.exit(1)

    if not source_directories:
        print(
            "[WARN] No Jam source folders were found in:\n"
            f"  {args.jam_root.resolve()}",
            flush=True,
        )

        return

    print("=" * 76)
    print("[INFO] Step 2: Demucs vocals.wav -> Raw Vocal MIDI / CSV / PNG")
    print(f"[INFO] Jam root: {args.jam_root.resolve()}")
    print(
        "[INFO] Candidates per onset: "
        f"{args.max_candidates_per_onset}"
    )
    print(
        "[INFO] Onset grouping tolerance: "
        f"{args.same_start_tolerance_ms:g} ms"
    )
    print(
        "[INFO] Minimum note duration: "
        f"{args.min_duration_ms:g} ms"
    )
    print(
        "[INFO] Pitch range: "
        f"{pretty_midi.note_number_to_name(args.pitch_min)} "
        f"(MIDI {args.pitch_min}) -> "
        f"{pretty_midi.note_number_to_name(args.pitch_max)} "
        f"(MIDI {args.pitch_max})"
    )
    print("[INFO] No BPM conversion. No rhythmic quantization.")
    print("[INFO] Rank 1 = blue by velocity.")
    print("[INFO] Rank 2 = fixed light purple.")
    print("=" * 76)

    success_count = 0

    for index, source_dir in enumerate(
        source_directories,
        start=1,
    ):
        print()
        print(
            f"[{index}/{len(source_directories)}] "
            f"Processing vocals: {source_dir.name}",
            flush=True,
        )

        try:
            note_count = process_one_source(
                source_dir=source_dir,
                args=args,
            )

            print(
                "  [OK] Completed. Retained vocal notes: "
                f"{note_count}",
                flush=True,
            )

            success_count += 1

        except Exception as error:
            print(
                f"  [FAIL] {error}",
                flush=True,
            )

    print()
    print("=" * 76)
    print(
        "[DONE] Successfully processed: "
        f"{success_count}/{len(source_directories)}",
        flush=True,
    )
    print()
    print("[OUTPUT]")
    print(
        "output_jam/by_source/<jam_name>/transcription/vocals/",
        flush=True,
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
