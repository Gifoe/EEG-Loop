# Residual channel diagnostic protocol

This descriptive R0 mechanism diagnostic uses the complete verified held-out archive. Convert loader amplitudes to microvolts by multiplying by 100. Estimate a separate mean waveform for each class, domain, channel, and time point from all evaluated class rows. Real and generated class counts must match. Record the counts and the finite-N shrinkage caused by estimating the means on these same rows.

Subtract that mean waveform, then subtract each window's channel mean over time. Compute sample covariance with denominator 999. Finite covariance entries remain valid, including exact zeros. Pearson correlation is undefined for a pair in a window when either channel's sample variance is at most 1e-12 microvolts squared. Reject nonfinite raw input.

Average finite covariance and valid Pearson estimates separately for each pair. Publish mean matrices, valid window counts, nearconstant channel counts, and nonfinite covariance counts. For each statistic, compute the mean absolute real/generated difference across the 120 unique upper-triangle channel pairs. A statistic's class discrepancy is undefined if any pair lacks an estimate in either domain. Its equal-six-class macro is undefined if any class discrepancy is undefined. Covariance is measured in microvolts squared, and correlation is unitless.

Use the exact 16 bipolar lead order from upstream data/preprocess_tuev.py and retain its source hash. Preserve the archive, index, and postprocessor hashes. This diagnostic introduces no change to the frozen quality gate.
