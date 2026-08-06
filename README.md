humming
python scripts/mp3_to_raw_top2_candidates

jam
python scripts/jam_step1_separate_stems.py
python scripts/jam_step2_transcribe_vocals.py
python scripts/jam_step3_export_fl_stems.py --device cuda
