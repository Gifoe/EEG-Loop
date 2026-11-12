# Official JET seed-0 training trace

The figure plots all 200 original epoch-mean training objectives and observed epoch durations from the completed TUEV run on a shared RTX A6000. Epoch labels are one-based. Values are unsmoothed and retain the precision of the training log.

The extraction checked epochs 0–199 in order and matched all 200 losses exactly against the independent training-completion record. Per-epoch durations exclude checkpoint writes and setup. This artifact describes training behavior and observed runtime.

Run totals are 9:48:38 by the trainer and 9:48:43 for the enclosing command. Logged objective starts at 1.2498 and ends at 0.2648. Observed epochs span 106–212 seconds.

Original training log SHA-256 is `0d8835fa63fb4e847abf9e7601ff19d4cfdefe5f1d5aa223401f9e737fa36a37`.

To render the supplied aggregate data, run `python plot_training.py .`. The script also supports extraction from the original remote log with `--log` and `--completion`. The PNG uses 300 dpi, SVG retains editable text, and PDF embeds DejaVu Sans with TrueType fonts. All source and output hashes are in artifact-manifest.json.
