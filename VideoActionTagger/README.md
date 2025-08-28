# VideoActionTagger

A macOS command-line tool that:
- Scans a folder of MP4 videos
- Runs a Core ML Action Classifier over sliding windows
- Merges window predictions into labeled time ranges
- Saves results to JSON and CSV per video
- Optionally writes Finder tags (extended attributes) to the file for Spotlight search

Notes:
- Embedding rich, structured metadata directly into MP4 (e.g., as custom atoms) is possible but not well-indexed by Spotlight. Finder tags are searchable; JSON/CSV sidecars are portable.
- Replace `ActionClassifier.mlmodel` with your trained model.

## Quick start
1. Place your `.mlmodel` at `Models/ActionClassifier.mlmodel`.
2. Build and run.
3. Provide the folder path via `--input`.

## Output
- `results/<video-basename>.json` with segments and probabilities
- `results/<video-basename>.csv` for spreadsheet use
- Optional: Finder tags summarizing top actions

## Assumptions
- Fixed window size and stride (configurable flags)
- Multi-label per window; merged by class with thresholds and min duration

## Future ideas
- Add audio classifier fusion
- Export EDL/CSV for NLEs
- Write QuickTime user data atoms if needed
