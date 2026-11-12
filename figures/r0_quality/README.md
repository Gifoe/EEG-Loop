# Seed-0 held-out quality figure

Panel a displays exact calibrated class-family means. Its sequential color scale spans 0–12, with values annotated to two decimals. Each value divides the native discrepancy by its corresponding frozen train/validation real–real scale, averages channels and primitives within the family, and averages the 200 fixed subset repetitions. The CSV preserves full precision.

Panel b takes the absolute values of the off-diagonal entries of each class/domain mean residual Pearson matrix and averages its 120 unique channel pairs. Absolute values are applied after the matrices have been averaged. The means of absolute per-window correlations are a different statistic.

The residual matrices use all evaluated samples. Real and generated domain counts match at 567, 4677, 1998, 329, 2204 and 19646. Their means are estimated from the same evaluated rows, with finite-N shrinkage. The calibrated subset counts are 92, 512, 380, 161, 512 and 512 per domain.

These are descriptive point summaries from one trained checkpoint. Overlapping subset repetitions describe sampling stability. The metadata retains exact input hashes and statistic definitions. Reproduce with Python plot_quality.py INPUT_DIRECTORY OUTPUT_DIRECTORY using the supplied aggregate quality packet.
