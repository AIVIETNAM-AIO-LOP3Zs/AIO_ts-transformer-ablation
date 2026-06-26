# Decoder Ablation Study Results

This report compiles the experimental findings of the Decoder architecture components on the ETT dataset forecasting task.

## Summary Table
```text

================================================================================
DECODER ABLATION STUDY - EMPIRICAL EVALUATION RESULTS
================================================================================

Variant                test_mse   test_mae      d MSE    d MSE %     Impact       Params
----------------------------------------------------------------------------------
baseline                 2.3559     1.0966    +0.0000      +0.0%   BASELINE      120,583
no-self-attention        2.2779     1.1912    -0.0780      -3.3%        LOW      120,583
no-causal-mask           2.5530     1.1715    +0.1971      +8.4%     MEDIUM      120,583
no-decoder               1.9991     1.1677    -0.3568     -15.1%        LOW    1,152,488
----------------------------------------------------------------------------------

[Interpretation]
  d MSE = MSE_variant - MSE_baseline
  A positive d MSE indicates performance degraded when removing the component.
  No Causal Masking (leakage) leading to poor generalization is expected.

[Rank] COMPONENT IMPORTANCE RANKING (most -> least critical to Decoder):
  1. Causal Masking (Future Leakage)     (d MSE = +0.1971)
  2. Decoder Self-Attention Layer        (d MSE = -0.0780)
  3. Complete Decoder Stack (Bypassed)   (d MSE = -0.3568)

```
