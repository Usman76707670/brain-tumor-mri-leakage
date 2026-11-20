# Leakage-controlled evaluation of CNN-transformer late fusion for four-class brain tumor MRI classification

Code, data splits, per-image predictions and result tables for the study by M. Usman Nasir (COMSATS University Islamabad, Sahiwal Campus).

The study trains ResNet101, Swin-V2-B and their equal-weight fusion on the public Sartaj Kaggle dataset (glioma, meningioma, pituitary tumor, no tumor) and asks how much of the reported accuracy depends on how the data are split. It recovers patient identities for most tumor images from the original Figshare dataset, builds a patient-aware split, measures the effect of near-duplicate images, and tests whether the no-tumor class can be recognised from image-source cues alone.

## Main results (patient-aware split, 623 test images, four seeds)

| Model | Accuracy % (mean ± SD) | 95% CI, patient clusters |
|---|---|---|
| Equal-weight fusion (primary, no TTA) | 92.58 ± 0.55 | 88.8 to 95.6 |
| Swin-V2-B | 92.17 ± 0.46 | 88.1 to 95.4 |
| EfficientNet-B0 | 92.22 ± 0.42 | 88.8 to 95.0 |
| ResNet101 | 91.65 ± 0.91 | 87.7 to 94.8 |
| ConvNeXt-T | 91.09 ± 0.80 | 86.3 to 94.7 |
| ResNet50 | 91.13 ± 0.75 | 86.7 to 94.8 |

The fusion is not shown to be better than Swin-V2-B (+0.40 points, 95% CI −0.29 to 1.08 with patient clusters resampled; −0.60 to 1.63 when the four seeds are also resampled) or than EfficientNet-B0. Averaging two Swin-V2-B models scored 92.94 ± 0.23%.

The same pipeline gave 97.97% on a random split and 95.69% on a split that grouped only near-duplicate images. The test sets of the three splits differ in size and class mix, so these are split-associated changes, not a pure estimate of leakage.

A classifier that sees only image size, file size and background statistics separates no-tumor from tumor test images with ROC AUC 0.978. Restricting the decision to the three tumor classes removes the no-tumor source from the comparison; the fusion then scores 91.78% (patient-cluster CI 87.2 to 95.4).

## Repository layout

| Folder | Contents |
|---|---|
| `code/train_seed.py` | Data loading, duplicate grouping, patient-aware split, models, training for one seed on one GPU |
| `code/statistics.py` | Reproduces every confidence interval, paired comparison and control from `predictions/` (no GPU needed) |
| `notebooks/experiments.ipynb` | The full Kaggle notebook: Figshare patient matching, splits, training on two GPUs, analyses, Grad-CAM, timing, external screening, pixel verification |
| `splits/` | Exact train/validation/test file lists. `split_patient.json` is the primary split; `split_grouped_t{0,3,5,8}.json` group near-duplicates only, at four hash thresholds |
| `data/image_table_hash_groups_splits.csv` | For every image: label, original folder, 64-bit perceptual hash, group IDs at each threshold, inferred Figshare patient ID, split membership |
| `data/sartaj_all_to_figshare_pid.csv` | Nearest Figshare image, hash distance, patient ID and Figshare label for every Sartaj image |
| `predictions/test_predictions_all_seeds.csv.gz` | Softmax outputs for every test image, model, seed and view (original and flipped) |
| `predictions/external_*_probs.npz` | Outputs on the two screened candidate external datasets |
| `results/internal/` | Ablations, paired tests, per-class metrics, calibration, screening intervals, thresholds, confounding, Grad-CAM checks, timing |
| `results/statistics/` | Output of `code/statistics.py` |
| `results/external_screening/` | Overlap, source and accuracy tables for BDNeuro-MRI and PMRAM |
| `results/verification/` | Pixel-level checks of hash matches |
| `config/` | Training configuration, timing protocol and the full package list of the Kaggle environment |
| `logs/` | Per-epoch training logs and kept epochs for every seed |

## How the patient-aware split was built

1. Every image of the Sartaj dataset (Training and Testing folders pooled, 3,264 images) was given a 64-bit perceptual hash (pHash, `imagehash`).
2. Each image was matched to its nearest image in the original Figshare dataset of Cheng et al., which records a patient ID per slice. A match within 5 bits gave the Sartaj image that patient ID (2,350 matches).
3. Images whose Sartaj label disagreed with the matched Figshare label were excluded (19 images).
4. Images were linked when their hashes differed by 5 bits or fewer or when they shared a patient ID, and linked images formed one group (715 groups).
5. Whole groups were assigned with `StratifiedGroupKFold` (seed 42): about 70% training, 10% validation, 20% test. The code stops if any group or patient appears in two partitions.

Result: 225 identified patients (164 / 20 / 41), none shared between partitions, and no test image within 5 bits of a training or validation image. 914 of the 3,245 images used have no verified identity: all 495 no-tumor images (Figshare has no healthy class) and 419 tumor images without a Figshare match. For these, separation from other images of the same patient cannot be verified, and the confidence intervals treat each as its own cluster.

## Verification of hash matches

Hash matches were checked against decoded pixels (`results/verification/`). A match counts as the same image when the pixel correlation after resizing both images to 256 x 256 is at least 0.98.

* Patient-ID matches: 2,324 of 2,350 are the same image. Of the other 26, 21 are tumor images with the same label as the Figshare image and correlations of 0.76 to 0.94; visual review shows neighbouring slices of the same tumor, so the patient ID is kept as an inferred identity. The remaining 5 are no-tumor images that matched Figshare tumor images at 4 bits; they are different scans, and their exclusion in step 3 was caused by false matches. They are excluded from every partition, so this cannot cause leakage. The only two ties between patients involve these false matches.
* The 14 pituitary images excluded for disagreeing with Figshare (meningioma) are the same images (correlation ≥ 0.9998), so they are genuine label disagreements between the two datasets.
* `Testing/glioma_tumor/image(48).jpg` is the same image (correlation ≥ 0.9985) as six images labelled meningioma. All seven are in the test partition. Removing it changes every model's accuracy by +0.15 points (`results/statistics/sensitivity_image48.csv`).

## External dataset screening

Two public four-class datasets described as hospital-collected were screened as possible external tests. Neither was used.

| Check | BDNeuro-MRI (DOI 10.17632/zwr4ntf94j, v9) | PMRAM (DOI 10.17632/m7w55sw88b, v1; original images only) |
|---|---|---|
| Images | 5,941 | 1,505 |
| 0-bit hash matches to Sartaj | 1,875 | 300 |
| of which byte-identical files | 1,611 | 0 (PMRAM was resized to 512 x 512) |
| of which decoded-pixel identical | 1,635 | 0 |
| of which pixel correlation ≥ 0.98 | 1,835 | 297 |
| Hash matches to Sartaj or Figshare within 5 bits | 2,135 (35.9%) | 402 (26.7%) |

Accuracy on the remaining images (95.23% and 96.83% for the fusion) is higher than on the internal test set, which is consistent with shared sources, so these figures are not reported as external validation. The no-tumor shortcut also appears in both datasets (background-only AUC 0.973 and 0.976).

## Reproducing the results

The statistics need only this repository:

```
pip install numpy pandas
python code/statistics.py
```

Training and the image-level analyses need the datasets and a GPU. Open `notebooks/experiments.ipynb` on Kaggle, attach the Sartaj dataset (and BDNeuro-MRI and PMRAM for the external cells), select GPU T4 x2 with internet on, and run the cells in order. Paths in the split files point to the Kaggle mount of the Sartaj dataset; match on the last three path parts (`Training/glioma_tumor/gg (1).jpg`) if the dataset is mounted elsewhere.

Training settings (also in `config/training_config.json`): 12 epochs, batch size 16, AdamW (weight decay 1e-4), learning rate 1e-4 (5e-5 for Swin-V2-B), cosine annealing without restarts, cross-entropy with label smoothing 0.1 and no class weighting, grayscale images copied to three channels, resize to 256 x 256 and centre crop to 224 x 224, training augmentation of horizontal flip (p = 0.5) and rotation up to 10 degrees, ImageNet normalisation, checkpoint kept at the highest validation accuracy (ties keep the earlier epoch), seeds 42 to 45 with deterministic cuDNN. Environment: PyTorch 2.10.0, torchvision 0.25.0, CUDA 12.8, Python 3.12, Tesla T4.

Model weights (about 2 GB for all seeds and models) are not stored here and are available from the author on request.

## Data sources

* Bhuvaji, S., Kadam, A., Bhumkar, P., Dedge, S., & Kanchan, S. (2020). *Brain tumor classification (MRI)* [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/1183165
* Cheng, J. (2017). *Brain tumor dataset* [Data set]. Figshare. https://doi.org/10.6084/m9.figshare.1512427.v5
* BDNeuro-MRI, Mendeley Data, https://doi.org/10.17632/zwr4ntf94j.9 (CC BY 4.0), and PMRAM, Mendeley Data, https://doi.org/10.17632/m7w55sw88b.1 (CC BY 4.0), used for screening only.

No images are redistributed in this repository; only file names, hashes, labels, inferred patient IDs and model outputs.

## Limitations

Patient identities are inferred by image matching, not supplied with the Sartaj dataset, and 914 images have none. Labels are dataset labels, not confirmed by histopathology in this study. All images are 2D slices of mixed planes and sequences, and the no-tumor class appears to come from different sources than the tumor classes (one no-tumor image is a CT scan). No independent external dataset was available. The results describe internal evaluation on one public collection and do not support clinical use.

## Licence

Code: MIT (see `LICENSE`). Derived tables and predictions: CC BY 4.0, with credit to the original datasets above.
