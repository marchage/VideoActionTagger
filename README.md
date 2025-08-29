# VideoActionTagger

macOS Swift CLI that scans a folder of MP4 videos, runs an Action Classification Core ML model over sliding windows, and merges predictions into labeled time ranges. It writes JSON/CSV sidecars and (optionally) Finder tags for Spotlight.

## Features
- Batch process a folder of `.mp4` files
- Sliding-window inference (configurable window/stride)
- Per-class thresholding and segment merging
- Outputs JSON and CSV per video
- Optional Finder tags (Spotlight searchable)

## Requirements
- macOS 13 or newer (Apple Silicon recommended)
- Swift 5.9+ (Xcode 15+ toolchain)
- A trained Core ML Action Classifier (`.mlmodel` or compiled `.mlmodelc`)

## Build
- From the package directory (`VideoActionTagger/`):
	- Build: `swift build -c debug`
	- The executable is produced at `.build/debug/video-action-tagger`

## Usage
Run the executable without args to see usage:

```
Usage: video-action-tagger --input /path/to/folder [--output out] [--model /path/to/ActionClassifier.mlmodel|.mlmodelc] [--window 3] [--stride 1] [--threshold 0.6] [--min-segment 1.5] [--no-tags]
```

Common examples:
- Process videos in a folder with a model and default settings:
	- `.build/debug/video-action-tagger --input /path/to/mp4s --model /path/to/ActionClassifier.mlmodel`
- Tweak windowing and thresholds:
	- `.build/debug/video-action-tagger --input /path/to/mp4s --model /path/to/ActionClassifier.mlmodel --window 4 --stride 1 --threshold 0.75 --min-segment 2`
- Disable Finder tags:
	- `.build/debug/video-action-tagger --input /path/to/mp4s --model /path/to/ActionClassifier.mlmodel --no-tags`

Notes:
- Only files directly inside `--input` are processed (non-recursive).
- Output defaults to `--input/results` unless `--output` is provided.

## Outputs
For each `<video>.mp4` you get:
- `results/<video>.json` — array of segments: `{ label, startSeconds, endSeconds, score }`
- `results/<video>.csv` — spreadsheet-friendly rows
- Finder tags on the video file with the unique set of predicted labels (unless `--no-tags`)

## Model setup (Create ML)
- Train an Action Classification model (Create ML “Action Classifier”) on short labeled clips per class, plus a “background” class if useful.
- Export the Core ML model and pass its path via `--model`.
- The code will compile `.mlmodel` to `.mlmodelc` at runtime if needed.

Important: `Sources/main.swift` includes a placeholder `ActionModel.predict(...)` method that returns dummy probabilities. Replace it with calls to your generated model class (from your `.mlmodel`) or wire it via Vision. Once connected, the tool will produce real predictions and segments.

## How it works (high level)
1. Windows: Samples frames over fixed windows (default 3s) with a stride (default 1s)
2. Inference: Calls the Core ML action classifier per window
3. Post-process: Thresholds, smooths, and merges consecutive windows per class into time segments
4. Serialize: Writes JSON/CSV and optional Finder tags

## Troubleshooting
- Build cache errors after moving the folder: clean and rebuild
	- `swift package clean` then delete `.build/` and rebuild
- “no tests found”: a test target isn’t configured yet; tests are TODO
- Empty outputs: ensure `--model` points to a valid Action Classifier and that you replaced the placeholder prediction code

## Roadmap
- Wire the action model to the generated Core ML interface
- Optional audio fusion (sound classifier) and late fusion
- Export EDL/XML for NLEs
- Optional QuickTime user data atoms for in-file metadata

## License
MIT (or your preferred license)
