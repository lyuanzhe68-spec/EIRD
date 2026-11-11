# EIRD: Evidence-Interventional Residual Diagnosis

Offline reproduction of the reported statistics for **Evidence-Interventional Residual Diagnosis for Frozen Stereo Matching**. This repository starts from saved per-frame correlations and diagnostic measurements. It does not contain the stereo network or execute model inference or fitting.

## Run the lightweight reproduction

Use Python 3.11 with the dependencies pinned in `requirements.txt`. The verified environment used Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1 and Pillow 12.3.0.

```sh
python -m pip install -r requirements.txt
python statistical_reproduction.py --out reproduced
```

Use an output directory without a prior result. The script writes `reproduction_results.json` and `reproduction_comparison.csv`. `PASS` means the computed result matches its comparison target. `FAIL` reports a numerical discrepancy. `NOT_REPRODUCED` documents a layer for which the required artifact was unavailable. The lightweight run produced **3,519 PASS, 0 FAIL and 4 NOT_REPRODUCED** (exit 0). Counts refer to ledger checks, not independent experiments.

The public-release check on 11 November 2026 also produced 3,519 PASS, 0 FAIL and 4 NOT_REPRODUCED using NumPy 2.1.3, SciPy 1.15.3 and Pillow 12.2.0.

## Scope and resampling

| Evidence | Computation |
| --- | --- |
| H1 and H2 | Equal-frame scores, paired differences and candidate-coordinate sensitivities from saved development observations. |
| V17.1 / V17.2 | Paired Val32 frame bootstrap, exact signed-rank tests, leave-one-frame-out summaries and error counts. Val32 is a reused development split. |
| V18 / V18.2 | External contrasts and original bootstrap intervals. KITTI clusters are 200 official training image IDs; drive membership is unverified. ETH3D resamples seven families, retaining all member views per draw. Both ETH3D mask definitions are retained. |
| V18.4 controls | RFF and Shuffle paired intervals and family summaries. The unfavorable ETH3D RFF comparisons remain in the data. |
| I3 / I4 | Recomputes differences from 160 saved response correlations and matching from 10,240 call-by-level random norm records. Original pixel response maps were not located. |

The optional pixel route accepts `--grid`, `--kitti` and `--eth` (see `--help`). It requires separately obtained local feature caches and disparity labels. These are outside this repository. The local full run replayed the α=1 controls and the original external bootstrap from pixel inputs. The portable run does not claim this layer. The archive also does not establish scene independence, intervention specificity or a causal mechanism.

`data/` preserves the small numeric source tables and parameter records used by the script. Machine-specific provenance paths were replaced with relative labels in four JSON files. The repository omits the manuscript, images, checkpoints, pixel caches, and historical repository content.

## License and third-party resources

The original reproduction code and author-produced numeric result tables in this repository are released under the MIT License. The license does not grant rights to third-party software, datasets, model weights, or imagery, which are not distributed here. Obtain FoundationStereo, Scene Flow, KITTI 2015, and ETH3D from their original providers and follow their terms.

## Paper and release

Associated manuscript: **Evidence-Interventional Residual Diagnosis for Frozen Stereo Matching with External Readout Transfer**, Yuanzhe Li. The manuscript is in preparation for submission; no publication or acceptance is claimed.

This release provides statistical reproduction from archived measurements. It does not include the EIRD inference implementation, pretrained backbone, feature caches, or the full experimental environment. The optional pixel route requires separately obtained caches and labels.

Release: **v1.0.0**, 11 November 2026. See [CITATION.cff](CITATION.cff) for citation metadata.
