# Universal Speech Content Factorization

We propose Universal Speech Content Factorization (USCF), a simple and invertible linear method for extracting a low-rank speech representation in which speaker timbre is suppressed while phonetic content is preserved. USCF extends Speech Content Factorization (SCF), a closed-set voice conversion method, to an open-set setting by learning a universal speech-to-content mapping via least-squares optimization and deriving speaker-specific transformations from only a few seconds of target speech. We show through embedding analysis that USCF effectively removes speaker-dependent variation. As a zero-shot voice conversion system, USCF achieves competitive intelligibility, naturalness, and speaker similarity compared to methods that require substantially more target-speaker data or additional neural training. Finally, we demonstrate that USCF features can serve as an alternative acoustic representation for text-to-speech, offering a linear, training-efficient substitute for timbre-prompted SSL-based systems. 

This codebase contains source code for the following:
- Content Factorization (following SCF proposed in LinearVC[^linearvc]).
- Voice Conversion (using SCF or USCF).
- TTS training using USCF features as acoustic target.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](license.md)

## Dependencies

Please refer to environment.yml. Make sure to install compatible version of PyTorch and cuVS.

## Directory

### WavLM feature extraction
LibriSpeech:
```
linearvc.egs.librispeech.extract_wavlm_libri \
    --librispeech_dir /path/to/librispeech/dev-clean \
    --output_dir /path/to/extract/features/dev-clean
```
TIMIT:
```
python -m linearvc.egs.timit.extract_wavlm_timit \
    --timit_dir /path/to/timit/TRAIN \
    --output_dir /path/to/extract/features/TRAIN
```

### Content Factorization
LibriSpeech:
```
python -m linearvc.egs.librispeech.experiments_libri \
    --subset dev-clean \
    --feats_dir /path/to/extracted/LibriSpeech/wavlm/feats \
    --rank 75 \
    --num_index 1 \
    --out_path_root /path/to/transforms/output/path
```
TIMIT:
```
python -m linearvc.egs.timit.experiments_timit \
    --subset TRAIN \
    --feats_dir /path/to/extracted/TIMIT/wavlm/feats \
    --wav_dir /path/to/TIMIT/root \
    --rank 75 \
    --num_index 1 \
    --out_path_root /path/to/transforms/output/path
```

### VC using kNN-VC, LinearVC, SCF, or USCF
LibriSpeech:
```
linearvc/egs/librispeech/convert_libri_linearvc.py
linearvc/egs/librispeech/convert_libri_cf.py
```
CommonVoice:
```
linearvc/egs/librispeech/convert_cv_knnvc.py
linearvc/egs/librispeech/convert_cv_cf.py
```

### Derive the Universal Speech-to-Content Mapping Matrix W
```
python -m linearvc.egs.librispeech.experiments_libri_inverse \
    --transform_root_dir \path\to\transform\root \
    --transform_type UTXSS
```

### TTS using USCF Features
Training:
```
python -m linearvc.cf_tts.train --config linearvc/cf_tts/config/config_v0.yaml
```
Inference:
```
linearvc/cf_tts/test.py
linearvc/cf_tts/test_bulk.py
```

### Evaluation Scripts
Scripts to be found in
```
linearvc/eval
```

## Interspeech Experiments and Results
Test scripts for Interspeech to be found in
```
linearvc/egs/librispeech/interspeech_test
```

### VC results:

| Method              | WER ↓    | UTMOS ↑ | Spk Sim ↑ | Spk EER ↓ | Tgt EER ↑ |
|---------------------|----------|---------|-----------|-----------|-----------|
| USCF, **W₁**        | 2.70%    | 2.805   | 0.524     | 1.62%     | 5.99%     |
| USCF, **W₂**        | 4.04%    | 2.519   | 0.557     | 1.31%     | 6.98%     |
| USCF, **W₃**        | 2.31%    | 2.826   | 0.420     | 4.92%     | 2.53%     |
| kNN-VC              | 3.16%    | 2.855   | **0.666** | **0.28%** | **17.66%**|
| LinearVC            | 2.69%    | 2.765   | 0.621     | 0.66%     | 12.09%    |
| SCF                 | **2.18%**| 2.886   | 0.603     | 0.79%     | 10.82%    |
| SCF, **W₁**         | 3.01%    | 2.859   | 0.604     | 0.68%     | 10.31%    |
| SeedVC              | 6.24%    | **3.173**| 0.532    | 2.13%     | 7.25%     |

### Comparison across different ranks:
| Rank | ASR WER ↓ | UTMOS-v2 ↑ | Spk Sim ↑ | Spk EER ↓ | Target EER ↑ |
|------|-----------|------------|-----------|-----------|--------------|
| 10   | 6.97%     | 1.885      | 0.435     | 2.25%     | 1.39%        |
| 20   | 3.98%     | 2.388      | 0.489     | 1.64%     | 3.24%        |
| 30   | 2.96%     | 2.607      | 0.513     | 1.46%     | 4.72%        |
| 50   | 2.69%     | 2.738      | 0.529     | 1.50%     | 5.94%        |
| 75   | 2.70%     | 2.805      | 0.524     | 1.62%     | 5.99%        |
| 100  | 2.77%     | 2.81       | 0.504     | 1.94%     | 5.29%        |

### Comparison across different number of frames available for speaker transformation matrix derivation:
| Num frames | ASR WER ↓ | UTMOS-v2 ↑ | Spk Sim ↑ | Spk EER ↓ | Target EER ↑ |
|------------|-----------|------------|-----------|-----------|--------------|
| 10000      | 2.28%     | 2.935      | 0.564     | 0.90%     | 7.86%        |
| 5000       | 2.42%     | 2.923      | 0.564     | 0.90%     | 7.86%        |
| 2000       | 2.47%     | 2.915      | 0.564     | 0.90%     | 7.86%        |
| 1000       | 2.51%     | 2.904      | 0.546     | 1.06%     | 7.16%        |
| 500        | 2.70%     | 2.805      | 0.524     | 1.62%     | 5.99%        |
| 200        | 4.94%     | 2.431      | 0.42      | 5.93%     | 2.71%        |
| 100        | 65.79%    | 1.544      | 0.22      | 23.01%    | 0.33%        |

## Acknowledgements

This codebase was developed on top of [LinearVC](https://github.com/kamperh/linearvc)[^linearvc] and [ZipVoice](https://github.com/k2-fsa/ZipVoice) [^zipvoice].

[^linearvc]: H. Kamper, B. van Niekerk, J. Za¨ıdi, and M.-A. Carbonneau, “LinearVC: Linear Transformations of Self-Supervised Features Through the Lens of Voice Conversion,” in Interspeech 2025, 2025, pp. 1398–1402. 

[^zipvoice]: H. Zhu, W. Kang, Z. Yao, L. Guo, F. Kuang, Z. Li, W. Zhuang, L. Lin, and D. Povey, “Zipvoice: Fast and high-quality zero-shot text-to-speech with flow matching,” 2025. [Online]. Available: https://arxiv.org/abs/2506.13053 