# Transcription Quality Comparison: iFlytek vs 11Labs

Generated: 2025-07-22 10:31:43

## Overall Performance Summary

| Dataset          | Service   | WER    | Accuracy   | WER (Aggressive)   | Accuracy (Aggressive)   | CER    |
|:-----------------|:----------|:-------|:-----------|:-------------------|:------------------------|:-------|
| 1476_JunKaiOng   | IFLYTEK   | 76.19% | 23.81%     | 79.88%             | 20.12%                  | 57.79% |
| 1476_JunKaiOng   | 11LABS    | 51.35% | 48.65%     | 49.60%             | 50.40%                  | 39.66% |
| 1985_JonathanKoh | IFLYTEK   | 53.76% | 46.24%     | 54.02%             | 45.98%                  | 37.69% |
| 1985_JonathanKoh | 11LABS    | 55.67% | 44.33%     | 45.56%             | 54.44%                  | 41.56% |

## Dataset: 1476_JunKaiOng

- Ground truth: 156 segments, 7359 characters

| Metric                | iFlytek   | 11Labs   |
|:----------------------|:----------|:---------|
| Segments              | 150       | 245      |
| Characters            | 5201      | 8657     |
| WER                   | 76.19%    | 51.35%   |
| Accuracy              | 23.81%    | 48.65%   |
| WER (Aggressive)      | 79.88%    | 49.60%   |
| Accuracy (Aggressive) | 20.12%    | 50.40%   |
| Character Error Rate  | 57.79%    | 39.66%   |

### Performance

- **11Labs** performs better on this dataset
- Accuracy advantage: **24.84%**

---

## Dataset: 1985_JonathanKoh

- Ground truth: 140 segments, 6801 characters

| Metric                | iFlytek   | 11Labs   |
|:----------------------|:----------|:---------|
| Segments              | 77        | 298      |
| Characters            | 6496      | 9203     |
| WER                   | 53.76%    | 55.67%   |
| Accuracy              | 46.24%    | 44.33%   |
| WER (Aggressive)      | 54.02%    | 45.56%   |
| Accuracy (Aggressive) | 45.98%    | 54.44%   |
| Character Error Rate  | 37.69%    | 41.56%   |

### Performance 

- **iFlytek** performs better on this dataset
- Accuracy advantage: **1.92%**

---

## Methodology

### Metrics Explained

- **WER (Word Error Rate)**: Percentage of words that differ from ground truth
- **Accuracy**: Percentage of words correctly transcribed (1 - WER)
- **WER (Aggressive)**: WER after removing filler words and normalizing equivalent terms
- **CER (Character Error Rate)**: Percentage of characters that differ from ground truth

### Normalization Levels

1. **Basic**: Lowercase, remove punctuation, normalize whitespace
2. **Aggressive**: Remove filler words (um, uh, like), normalize equivalent terms (ok→okay, yeah→yes)

### Why Accuracy Matters More Than WER

WER can be misleadingly high due to:
- Different handling of filler words
- Formatting and punctuation differences
- Equivalent terms counted as errors
- Mixed language tokenization challenges

Higher accuracy percentage = better transcription quality
