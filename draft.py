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

# Must be set before importing pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import pretty_midi
from basic_pitch.inference import predict
from music21 import meter, metadata, note as music21_note, stream, tempo


AUDIO_EXTENSIONS = {".mp3"}

OUTPUT_FOLDERS = {
    "midi": "01_midi",
    "musicxml": "02_musicxml",
    "notes_csv": "03_notes_transposed_csv",
    "pianoroll_png": "04_pianoroll_png",
    "pianoroll_grid_csv": "05_pianoroll_grid_csv",
}

CSV_FIELD_ORDER = [
    "note_index",
    "note_name",
    "midi_pitch",
    "velocity",
    "source_start_seconds",
    "source_end_seconds",
    "source_duration_seconds",
    "draft_start_quarter_length",
    "draft_duration_quarter_length",
    "draft_start_beats",
    "draft_duration_beats",
    "bpm_used",
    "time_signature",
    "subdivision_per_quarter",
]


@dataclass
class DraftNote:
    """One note after converting raw seconds into quantized musical timing."""

    source_start_seconds: float
    source_end_seconds: float
    source_duration_seconds: float
    pitch: int
    note_name: str
    velocity: int
    start_quarter_length: float
    duration_quarter_length: float


def parse_time_signature(value: str) -> tuple[int, int]:
    """Parse strings such as 4/4, 3/4, 6/8."""
    match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", value.strip())

    if not match:
        raise argparse.ArgumentTypeError(
            "拍号格式应类似：4/4、3/4、6/8。"
        )

    numerator = int(match.group(1))
    denominator = int(match.group(2))

    if numerator <= 0:
        raise argparse.ArgumentTypeError("拍号分子必须大于 0。")

    if denominator not in {1, 2, 4, 8, 16, 32}:
        raise argparse.ArgumentTypeError(
            "拍号分母只能为 1、2、4、8、16、32。"
        )

    return numerator, denominator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read MP3 files, transcribe melody with Basic Pitch, "
            "then export MIDI, MusicXML, CSV, PNG, and piano-roll CSV."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="MP3 输入目录。默认：input",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="输出根目录。默认：output",
    )

    parser.add_argument(
        "--bpm",
        type=float,
        default=80.0,
        help="用于乐谱量化的候选 BPM。默认：80",
    )

    parser.add_argument(
        "--time-signature",
        type=parse_time_signature,
        default=(4, 4),
        help="拍号，例如 4/4、3/4、6/8。默认：4/4",
    )

    parser.add_argument(
        "--subdivision",
        type=int,
        choices=[1, 2, 4, 8],
        default=2,
        help=(
            "每个四分音符划分几格："
            "1=四分音符，2=八分音符，4=十六分音符，8=三十二分音符。"
            "默认：2"
        ),
    )

    parser.add_argument(
        "--min-duration-ms",
        type=float,
        default=45.0,
        help="忽略短于该长度的识别音符。默认：45 毫秒",
    )

    parser.add_argument(
        "--keep-overlaps",
        action="store_true",
        help=(
            "保留重叠音符。默认会尽量整理为单旋律线，"
            "适合人声哼唱。"
        ),
    )

    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.5,
        help="Basic Pitch 起音识别阈值。默认：0.5",
    )

    parser.add_argument(
        "--frame-threshold",
        type=float,
        default=0.3,
        help="Basic Pitch 持续音识别阈值。默认：0.3",
    )

    args = parser.parse_args()

    if args.bpm <= 0:
        parser.error("--bpm 必须大于 0。")

    if args.min_duration_ms < 0:
        parser.error("--min-duration-ms 不可小于 0。")

    if not 0 <= args.onset_threshold <= 1:
        parser.error("--onset-threshold 必须在 0 到 1 之间。")

    if not 0 <= args.frame_threshold <= 1:
        parser.error("--frame-threshold 必须在 0 到 1 之间。")

    return args


def collect_audio_files(input_dir: Path) -> list[Path]:
    """Collect MP3 files directly inside the input directory."""
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def round_to_grid(value: float, grid: float) -> float:
    """Round a value to the nearest quantization grid."""
    return round(value / grid) * grid


def format_bpm_for_filename(bpm: float) -> str:
    """80.5 -> 80_5, suitable for filenames."""
    return f"{bpm:g}".replace(".", "_")


def get_basic_pitch_notes(
    source_path: Path,
    onset_threshold: float,
    frame_threshold: float,
    minimum_note_length_ms: float,
) -> list[pretty_midi.Note]:
    """Run Basic Pitch on one MP3 and return non-drum MIDI notes."""

    print(f"  [STEP 1/5] Basic Pitch 转写：{source_path.name}")

    _, midi_data, _ = predict(
        str(source_path),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length_ms,
    )

    notes: list[pretty_midi.Note] = []

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue

        notes.extend(instrument.notes)

    return sorted(
        notes,
        key=lambda item: (item.start, item.pitch, item.end),
    )


def make_monophonic(notes: list[pretty_midi.Note]) -> list[pretty_midi.Note]:
    """Try to turn Basic Pitch output into one melody line.

    Humming is usually monophonic, but pitch transcription can occasionally
    detect simultaneous or overlapping notes. This function keeps the stronger
    / longer note at the same onset and trims overlaps.
    """
    if not notes:
        return []

    same_start_tolerance = 0.025
    grouped_notes: list[pretty_midi.Note] = []

    index = 0

    while index < len(notes):
        current = notes[index]
        same_start_group = [current]
        index += 1

        while (
            index < len(notes)
            and abs(notes[index].start - current.start)
            <= same_start_tolerance
        ):
            same_start_group.append(notes[index])
            index += 1

        best_note = max(
            same_start_group,
            key=lambda item: (
                item.end - item.start,
                item.velocity,
            ),
        )

        grouped_notes.append(
            pretty_midi.Note(
                velocity=best_note.velocity,
                pitch=best_note.pitch,
                start=best_note.start,
                end=best_note.end,
            )
        )

    result: list[pretty_midi.Note] = []

    for current_note in grouped_notes:
        if not result:
            result.append(current_note)
            continue

        previous_note = result[-1]

        if current_note.start < previous_note.end:
            previous_note.end = current_note.start

            if previous_note.end - previous_note.start < 0.02:
                result.pop()

        result.append(current_note)

    return result


def convert_to_draft_notes(
    raw_notes: list[pretty_midi.Note],
    bpm: float,
    subdivision: int,
) -> list[DraftNote]:
    """Convert raw seconds into quantized quarter-note timing."""
    if not raw_notes:
        return []

    seconds_per_quarter = 60.0 / bpm
    grid_quarter_length = 1.0 / subdivision

    first_start_seconds = raw_notes[0].start
    draft_notes: list[DraftNote] = []

    for raw_note in raw_notes:
        source_duration_seconds = raw_note.end - raw_note.start
        relative_start_seconds = raw_note.start - first_start_seconds

        raw_start_quarter_length = (
            relative_start_seconds / seconds_per_quarter
        )
        raw_duration_quarter_length = (
            source_duration_seconds / seconds_per_quarter
        )

        quantized_start = round_to_grid(
            raw_start_quarter_length,
            grid_quarter_length,
        )

        quantized_duration = round_to_grid(
            raw_duration_quarter_length,
            grid_quarter_length,
        )

        quantized_duration = max(
            grid_quarter_length,
            quantized_duration,
        )

        draft_notes.append(
            DraftNote(
                source_start_seconds=raw_note.start,
                source_end_seconds=raw_note.end,
                source_duration_seconds=source_duration_seconds,
                pitch=raw_note.pitch,
                note_name=pretty_midi.note_number_to_name(raw_note.pitch),
                velocity=raw_note.velocity,
                start_quarter_length=quantized_start,
                duration_quarter_length=quantized_duration,
            )
        )

    return sorted(
        draft_notes,
        key=lambda item: (
            item.start_quarter_length,
            item.pitch,
        ),
    )

def make_quantized_monophonic(
    draft_notes: list[DraftNote],
) -> list[DraftNote]:
    """Ensure only one note is active at every quantized time.

    This runs AFTER rhythmic quantization.

    Why it is needed:
    Two raw notes with slightly different start times can be quantized onto
    the same rhythmic grid point. That can create unwanted chords even when
    the original humming was monophonic.

    Rules:
    1. If multiple notes start at the same quantized position, keep one.
    2. Prefer longer notes.
    3. If duration is equal, prefer higher velocity.
    4. If still equal, prefer higher pitch.
    5. If a note overlaps the following note, shorten it so the melody
       remains strictly monophonic.
    """
    if not draft_notes:
        return []

    notes_sorted = sorted(
        draft_notes,
        key=lambda item: (
            item.start_quarter_length,
            item.pitch,
        ),
    )

    # Step 1:
    # At each identical quantized start position, retain only one note.
    one_note_per_start: list[DraftNote] = []

    index = 0

    while index < len(notes_sorted):
        current_start = notes_sorted[index].start_quarter_length
        same_start_notes = [notes_sorted[index]]
        index += 1

        while (
            index < len(notes_sorted)
            and notes_sorted[index].start_quarter_length == current_start
        ):
            same_start_notes.append(notes_sorted[index])
            index += 1

        best_note = max(
            same_start_notes,
            key=lambda item: (
                item.duration_quarter_length,
                item.velocity,
                item.pitch,
            ),
        )

        one_note_per_start.append(best_note)

    # Step 2:
    # Remove overlap between consecutive notes by trimming the earlier note.
    result: list[DraftNote] = []

    for note_index, current_note in enumerate(one_note_per_start):
        is_last_note = note_index == len(one_note_per_start) - 1

        if is_last_note:
            result.append(current_note)
            continue

        next_note = one_note_per_start[note_index + 1]

        current_end = (
            current_note.start_quarter_length
            + current_note.duration_quarter_length
        )

        # No overlap: keep original quantized duration.
        if current_end <= next_note.start_quarter_length:
            result.append(current_note)
            continue

        trimmed_duration = (
            next_note.start_quarter_length
            - current_note.start_quarter_length
        )

        # If the next note starts at exactly the same time, current_note
        # should disappear. Normally this was already handled in Step 1.
        if trimmed_duration <= 0:
            continue

        result.append(
            DraftNote(
                source_start_seconds=current_note.source_start_seconds,
                source_end_seconds=current_note.source_end_seconds,
                source_duration_seconds=current_note.source_duration_seconds,
                pitch=current_note.pitch,
                note_name=current_note.note_name,
                velocity=current_note.velocity,
                start_quarter_length=current_note.start_quarter_length,
                duration_quarter_length=trimmed_duration,
            )
        )

    return result



def build_music21_score(
    source_name: str,
    draft_notes: list[DraftNote],
    bpm: float,
    time_signature: tuple[int, int],
) -> stream.Score:
    """Create a score suitable for MIDI and MusicXML export."""
    numerator, denominator = time_signature

    score = stream.Score()
    score.metadata = metadata.Metadata()

    # Metadata can include Chinese safely; it is not used by matplotlib.
    score.metadata.title = source_name
    score.metadata.composer = "Draft generated from MP3 humming"

    part = stream.Part()
    part.partName = "Melody"

    part.insert(0, meter.TimeSignature(f"{numerator}/{denominator}"))
    part.insert(0, tempo.MetronomeMark(number=bpm))

    for draft_note in draft_notes:
        output_note = music21_note.Note()
        output_note.pitch.midi = draft_note.pitch
        output_note.duration.quarterLength = (
            draft_note.duration_quarter_length
        )
        output_note.volume.velocity = draft_note.velocity

        part.insert(
            draft_note.start_quarter_length,
            output_note,
        )

    part.makeMeasures(inPlace=True)
    score.insert(0, part)

    return score


def draft_notes_to_csv_rows(
    draft_notes: list[DraftNote],
    bpm: float,
    subdivision: int,
    time_signature: tuple[int, int],
) -> list[dict[str, str | int]]:
    """Convert draft notes into rows before writing a transposed CSV."""
    numerator, denominator = time_signature
    time_signature_text = f"{numerator}/{denominator}"

    rows: list[dict[str, str | int]] = []

    for index, draft_note in enumerate(draft_notes, start=1):
        rows.append(
            {
                "note_index": index,
                "note_name": draft_note.note_name,
                "midi_pitch": draft_note.pitch,
                "velocity": draft_note.velocity,
                "source_start_seconds": (
                    f"{draft_note.source_start_seconds:.6f}"
                ),
                "source_end_seconds": (
                    f"{draft_note.source_end_seconds:.6f}"
                ),
                "source_duration_seconds": (
                    f"{draft_note.source_duration_seconds:.6f}"
                ),
                "draft_start_quarter_length": (
                    f"{draft_note.start_quarter_length:.3f}"
                ),
                "draft_duration_quarter_length": (
                    f"{draft_note.duration_quarter_length:.3f}"
                ),
                "draft_start_beats": (
                    f"{draft_note.start_quarter_length:.3f}"
                ),
                "draft_duration_beats": (
                    f"{draft_note.duration_quarter_length:.3f}"
                ),
                "bpm_used": f"{bpm:.2f}",
                "time_signature": time_signature_text,
                "subdivision_per_quarter": subdivision,
            }
        )

    return rows


def write_transposed_notes_csv(
    output_path: Path,
    rows: list[dict[str, str | int]],
) -> None:
    """Write one note per column and one property per row."""
    note_headers = [
        f"note_{index:03d}"
        for index in range(1, len(rows) + 1)
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(["field", *note_headers])

        for field_name in CSV_FIELD_ORDER:
            writer.writerow(
                [
                    field_name,
                    *[row.get(field_name, "") for row in rows],
                ]
            )


def write_pianoroll_grid_csv(
    output_path: Path,
    draft_notes: list[DraftNote],
    bpm: float,
    subdivision: int,
) -> None:
    """Write piano-roll-like CSV.

    Horizontal axis: quantized time in seconds.
    Vertical axis: MIDI pitch / note name.
    Cell value: MIDI velocity when that pitch is active in the time slot.
    """
    if not draft_notes:
        raise ValueError("No draft notes available for piano-roll CSV.")

    seconds_per_quarter = 60.0 / bpm
    grid_quarter_length = 1.0 / subdivision

    min_pitch = max(0, min(note.pitch for note in draft_notes) - 2)
    max_pitch = min(127, max(note.pitch for note in draft_notes) + 2)

    end_quarter_length = max(
        note.start_quarter_length + note.duration_quarter_length
        for note in draft_notes
    )

    slot_count = max(
        1,
        math.ceil(end_quarter_length / grid_quarter_length),
    )

    time_headers: list[str] = []

    for slot_index in range(slot_count):
        quarter_length = slot_index * grid_quarter_length
        seconds = quarter_length * seconds_per_quarter
        time_headers.append(f"{seconds:.3f}s")

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(["pitch_or_time", *time_headers])

        for pitch in range(max_pitch, min_pitch - 1, -1):
            row_cells: list[str] = []

            for slot_index in range(slot_count):
                slot_start = slot_index * grid_quarter_length
                slot_end = slot_start + grid_quarter_length

                active_notes = [
                    note
                    for note in draft_notes
                    if note.pitch == pitch
                    and note.start_quarter_length < slot_end
                    and (
                        note.start_quarter_length
                        + note.duration_quarter_length
                    ) > slot_start
                ]

                if active_notes:
                    velocity = max(
                        note.velocity
                        for note in active_notes
                    )
                    row_cells.append(str(velocity))
                else:
                    row_cells.append("")

            pitch_name = pretty_midi.note_number_to_name(pitch)

            writer.writerow(
                [
                    f"{pitch_name} (MIDI {pitch})",
                    *row_cells,
                ]
            )


def render_draft_piano_roll(
    output_path: Path,
    draft_notes: list[DraftNote],
    bpm: float,
    time_signature: tuple[int, int],
) -> None:
    """Render quantized draft notes as a PNG piano roll.

    All chart text is English / ASCII only, intentionally avoiding Chinese
    glyph warnings from DejaVu Sans.
    """
    if not draft_notes:
        raise ValueError("No draft notes available for piano-roll PNG.")

    min_pitch = min(note.pitch for note in draft_notes)
    max_pitch = max(note.pitch for note in draft_notes)

    pitch_low = max(0, min_pitch - 2)
    pitch_high = min(127, max_pitch + 2)

    end_beat = max(
        note.start_quarter_length + note.duration_quarter_length
        for note in draft_notes
    )

    figure_width = max(10, min(26, 7 + end_beat * 0.8))
    figure_height = max(
        4,
        min(13, 3 + (pitch_high - pitch_low) * 0.34),
    )

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    patches = []
    velocities = []

    for note in draft_notes:
        patches.append(
            Rectangle(
                (
                    note.start_quarter_length,
                    note.pitch - 0.36,
                ),
                note.duration_quarter_length,
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

    ax.set_xlim(-0.15, end_beat + 0.3)
    ax.set_ylim(pitch_low - 0.6, pitch_high + 0.6)

    ax.set_xlabel("Quantized timing (quarter-note beats)")
    ax.set_ylabel("Pitch")
    ax.set_title(
        f"Piano Roll | BPM {bpm:g} | "
        f"Time Signature {time_signature[0]}/{time_signature[1]}"
    )

    for pitch in range(pitch_low, pitch_high + 1):
        ax.axhline(
            pitch,
            color="#d9d9d9",
            linewidth=0.5,
            zorder=0,
        )

    for beat in range(0, int(math.ceil(end_beat)) + 2):
        ax.axvline(
            beat,
            color="#d3dce5",
            linewidth=0.8,
            zorder=0,
        )

    ticks = list(range(pitch_low, pitch_high + 1))

    ax.set_yticks(ticks)
    ax.set_yticklabels(
        [
            pretty_midi.note_number_to_name(pitch)
            for pitch in ticks
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
    bpm: float,
    time_signature: tuple[int, int],
    subdivision: int,
) -> dict[str, dict[str, Path]]:
    """Create paths for per-source outputs and aggregate output folders."""
    numerator, denominator = time_signature
    bpm_label = format_bpm_for_filename(bpm)

    base_name = (
        f"{source_path.stem}_draft_{bpm_label}bpm_"
        f"{numerator}-{denominator}_grid{subdivision}"
    )

    source_dir = output_root / "by_source" / source_path.stem

    aggregate_dirs = {
        key: output_root / folder_name
        for key, folder_name in OUTPUT_FOLDERS.items()
    }

    source_dir.mkdir(parents=True, exist_ok=True)

    for directory in aggregate_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "source": {
            "midi": source_dir / f"{base_name}.mid",
            "musicxml": source_dir / f"{base_name}.musicxml",
            "notes_csv": (
                source_dir
                / f"{base_name}_notes_transposed.csv"
            ),
            "pianoroll_png": (
                source_dir
                / f"{base_name}_pianoroll.png"
            ),
            "pianoroll_grid_csv": (
                source_dir
                / f"{base_name}_pianoroll_grid.csv"
            ),
        },
        "aggregate": {
            "midi": (
                aggregate_dirs["midi"]
                / f"{base_name}.mid"
            ),
            "musicxml": (
                aggregate_dirs["musicxml"]
                / f"{base_name}.musicxml"
            ),
            "notes_csv": (
                aggregate_dirs["notes_csv"]
                / f"{base_name}_notes_transposed.csv"
            ),
            "pianoroll_png": (
                aggregate_dirs["pianoroll_png"]
                / f"{base_name}_pianoroll.png"
            ),
            "pianoroll_grid_csv": (
                aggregate_dirs["pianoroll_grid_csv"]
                / f"{base_name}_pianoroll_grid.csv"
            ),
        },
    }


def copy_to_aggregate(
    source_paths: dict[str, Path],
    aggregate_paths: dict[str, Path],
) -> None:
    """Copy the five source-folder files into the five aggregate folders."""
    for key, source_file in source_paths.items():
        destination_file = aggregate_paths[key]
        shutil.copy2(source_file, destination_file)


def process_one_mp3(
    source_path: Path,
    output_root: Path,
    bpm: float,
    time_signature: tuple[int, int],
    subdivision: int,
    min_duration_ms: float,
    keep_overlaps: bool,
    onset_threshold: float,
    frame_threshold: float,
) -> int:
    """Process one MP3 and create all required outputs."""

    raw_notes = get_basic_pitch_notes(
        source_path=source_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length_ms=min_duration_ms,
    )

    if not raw_notes:
        raise ValueError("Basic Pitch 没有识别到可用音符。")

    if not keep_overlaps:
        raw_notes = make_monophonic(raw_notes)

    if not raw_notes:
        raise ValueError("单旋律整理后没有保留音符。")

    print("  [STEP 2/5] 节奏量化并生成乐谱草稿")

  draft_notes = convert_to_draft_notes(
      raw_notes=raw_notes,
      bpm=bpm,
      subdivision=subdivision,
  )
  
  if not draft_notes:
      raise ValueError("量化后没有可用音符。")
  
  # Important:
  # Raw-note monophonic cleanup is not enough, because quantization can move
  # two different raw note starts onto the same rhythmic grid position.
  #
  # By default, keep the result strictly monophonic after quantization.
  # Use --keep-overlaps if you intentionally want chords / overlapping notes.
  if not keep_overlaps:
      draft_notes = make_quantized_monophonic(draft_notes)
  
  if not draft_notes:
      raise ValueError("单声部量化整理后没有可用音符。")


    if not draft_notes:
        raise ValueError("量化后没有可用音符。")

    output_paths = build_output_paths(
        output_root=output_root,
        source_path=source_path,
        bpm=bpm,
        time_signature=time_signature,
        subdivision=subdivision,
    )

    score = build_music21_score(
        source_name=source_path.name,
        draft_notes=draft_notes,
        bpm=bpm,
        time_signature=time_signature,
    )

    print("  [STEP 3/5] 写入 MIDI 和 MusicXML")

    score.write(
        "midi",
        fp=str(output_paths["source"]["midi"]),
    )

    score.write(
        "musicxml",
        fp=str(output_paths["source"]["musicxml"]),
    )

    csv_rows = draft_notes_to_csv_rows(
        draft_notes=draft_notes,
        bpm=bpm,
        subdivision=subdivision,
        time_signature=time_signature,
    )

    print("  [STEP 4/5] 写入两份 CSV")

    write_transposed_notes_csv(
        output_path=output_paths["source"]["notes_csv"],
        rows=csv_rows,
    )

    write_pianoroll_grid_csv(
        output_path=output_paths["source"]["pianoroll_grid_csv"],
        draft_notes=draft_notes,
        bpm=bpm,
        subdivision=subdivision,
    )

    print("  [STEP 5/5] 生成钢琴卷帘 PNG")

    render_draft_piano_roll(
        output_path=output_paths["source"]["pianoroll_png"],
        draft_notes=draft_notes,
        bpm=bpm,
        time_signature=time_signature,
    )

    copy_to_aggregate(
        source_paths=output_paths["source"],
        aggregate_paths=output_paths["aggregate"],
    )

    return len(draft_notes)


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        print(f"[ERROR] 输入目录不存在：{args.input_dir.resolve()}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = collect_audio_files(args.input_dir)

    if not audio_files:
        print(f"[WARN] 没有找到 MP3 文件：{args.input_dir.resolve()}")
        return

    numerator, denominator = args.time_signature

    print("=" * 62)
    print("[INFO] MP3 -> MIDI / MusicXML / CSV / PNG")
    print(f"[INFO] 输入目录：{args.input_dir.resolve()}")
    print(f"[INFO] 输出目录：{args.output_dir.resolve()}")
    print(f"[INFO] BPM：{args.bpm}")
    print(f"[INFO] 拍号：{numerator}/{denominator}")
    print(f"[INFO] 节奏网格：每个四分音符 {args.subdivision} 格")
    print("[INFO] 单文件输出：output/by_source/原文件名/")
    print("[INFO] 汇总输出目录：")
    print("[INFO]   output/01_midi/")
    print("[INFO]   output/02_musicxml/")
    print("[INFO]   output/03_notes_transposed_csv/")
    print("[INFO]   output/04_pianoroll_png/")
    print("[INFO]   output/05_pianoroll_grid_csv/")
    print("=" * 62)

    success_count = 0

    for index, source_path in enumerate(audio_files, start=1):
        print()
        print(
            f"[{index}/{len(audio_files)}] 处理文件："
            f"{source_path.name}"
        )

        try:
            note_count = process_one_mp3(
                source_path=source_path,
                output_root=args.output_dir,
                bpm=args.bpm,
                time_signature=args.time_signature,
                subdivision=args.subdivision,
                min_duration_ms=args.min_duration_ms,
                keep_overlaps=args.keep_overlaps,
                onset_threshold=args.onset_threshold,
                frame_threshold=args.frame_threshold,
            )

            print(f"  [OK] 完成，共生成 {note_count} 个草稿音符。")
            success_count += 1

        except Exception as error:
            print(f"  [FAIL] 处理失败：{error}")

    print()
    print("=" * 62)
    print(
        f"[DONE] 完成：{success_count}/{len(audio_files)} 个 MP3 文件。"
    )
    print("=" * 62)


if __name__ == "__main__":
    main()
