# PruDensNet: brain tumour MRI classification

Code and executed notebooks for "PruDensNet: a parameter efficient depthwise
separable CNN for MRI-based brain tumor classification", Frontiers in Medicine
2026, doi:10.3389/fmed.2026.1743015.

The model is a small CNN with depthwise separable convolutions, squeeze-excite
and spatial attention, trained from scratch on the public Kaggle
brain-tumor-mri-dataset (masoudnickparvar) for the 4-class task: glioma,
meningioma, pituitary, no tumour. The paper reports 96.05% test accuracy and
97.27% validation accuracy.

## Read this first

I would rather you learn these four things from me than find them yourself.

- The real trainable parameter count is 1,130,382. The published figure of
  1,461,587 (about 1.46M) includes a ParamPad layer: 331,205 trainable weights
  whose output is multiplied by 0.0, so they have zero gradient and never do
  anything. The paper does disclose this, in section 3.4.7 (equations 24 and
  29), and Algorithm 2 calls it a no-op layer.
- The published 96.05/97.27 came from a random file-level 80/10/10 split
  (seed 42) of 4,785 images from the dataset's Training/ folder, after
  per-class aHash deduplication. The Testing/ folder was never read in that
  run, even though the Methods section says Testing/ was the held-out test set
  with a 90/10 train/val split of Training/.
- You can check this against the paper itself. The published per-class table
  has a test support of 481 images, which is the random split's test set. The
  dataset's actual Testing/ folder has 1,311 images.
- Authorship: Arman, Samin, Islam, Arnob, Islam, Mamoon. I am the second
  author. The CRediT statement lists me for Conceptualization, Methodology,
  Software and Writing (original draft). I designed the architecture and wrote
  all the code in notebooks/.

## Which notebook made which number

| Notebook | What it produced | The split it actually used |
|---|---|---|
| `test-acc-96-05-14-and-validation-acc-97-27.ipynb` | The headline numbers: test 96.05% and validation 97.27%, both from SWA-averaged weights with 5x test-time augmentation. Also the per-class precision/recall/F1 tables (test support 481) and the confusion matrices | Random file-level 80/10/10 (`splitfolders.ratio`, seed 42) over 4,785 deduplicated images from Training/ only. Train 3,827 / val 477 / test 481 |
| `00-`, `000-`, `02-`, `05-`, `06-`, `brain-tumor-detection-mri-dataset.ipynb` | The baseline numbers (MaxViT, ViT, Swin V2, ConvNeXtV2, EfficientNet, LeViT, NFNet, RegNetY and the rest of the 29-model pool). All of these are validation accuracies | Random file-level 90/10 train/val (4,321 / 481). These runs have no test set |
| Cell 12 of the headline notebook | The complexity table: params, MACs, FLOPs, peak GPU memory, median latency on a Tesla P100 | n/a |
| `Expected Calibration Error (ECE).ipynb` | The ECE and Brier score table | reuses the headline run's predictions |

## The two questions a careful reader will have

### 1. The parameter count is padded

The real model is 1,130,382 trainable parameters. ParamPad holds the other
331,205: a vector of weights initialised to zero, multiplied by 0.0 in the
forward pass, so their gradient is zero and they never move. The layer exists
to raise the reported trainable count to a fixed target.

The same pad was applied to every baseline. The 29 comparison models are
pretrained backbones whose trainable heads range from about 50k to 1.6M
parameters, and each head was padded up to 1,461,587 so the trainable-params
column is equal across the table. What the pad cannot do is equalise capacity.
Dead weights are not capacity, and the baselines' total sizes, counting their
frozen pretrained backbones, run from 2.4M to 70M parameters. If you want one
number for this model's size, it is 1,130,382.

### 2. The split in the paper is not the split in the code

The Methods (section 3.2.3, Table 1, Algorithm 1) say the dataset's predefined
Testing/ folder is the held-out test set and Training/ is split 90/10. No
notebook in this repository does that. The notebook that produced the published
numbers resolved its data root directly to Training/, deduplicated it, and
split it randomly 80/10/10 at file level. Testing/ was never read.

The paper's own tables match the code, not the Methods text: the published
test support is 481, the random split's test set, while the real Testing/
folder holds 1,311 images.

What that means in practice:

- The published test set is a random tenth of the deduplicated Training/ pool,
  drawn with seed 42, not the dataset's curated test partition.
- The aHash deduplication ran first, so exact and near-exact duplicates could
  not sit on both sides of the split. That removes the most obvious leakage
  mode in this dataset. It cannot catch different slices from the same patient,
  because the dataset has no patient identifiers. The Limitations section of
  the paper says this, and on that point it is accurate.
- The baselines were measured on a different random split (90/10, no test set)
  and are validation numbers, so the comparison table puts our test accuracy
  next to their validation accuracy, under pools that are not identical.

## Other places where the paper and the code disagree

I found these by reading the executed notebook against the published Methods.
Where they differ, the code is what actually produced the numbers.

| Item | Paper says | Code did |
|---|---|---|
| Dropout (head) | 0.35 | 0.25 |
| Base LR | 3e-4 | 2e-4 |
| Gradient clipping | global norm 1.0 | global norm 0.6 |
| Label smoothing | dynamic, 0.05 / 0.03 / 0 by epoch phase | fixed 0.02 |
| MixUp | Beta(0.3, 0.3), prob up to 0.25 | Beta(0.25, 0.25), prob up to 0.10 |
| Augmentation | rot ±10°, shift 0.08, shear 6°, zoom [0.8, 1.2], brightness [0.92, 1.10] | rot ±6°, shift 0.04, shear 2°, zoom 0.10, brightness [0.97, 1.04] |
| Class weights | w = N / (K·n_k) | same formula, then clipped to [0.85, 1.20] |
| Stopping | early stopping on accuracy, patience 12, best checkpoint restored | fixed 50 epochs, no early stopping, SWA average of the last 16 epochs |
| Evaluation model | reload best checkpoint (Algorithm 3) | the SWA-averaged final weights |
| Precision | mixed_float16 policy | not enabled, float32 throughout |
| Seeds | results across multiple random seeds, mean and std | the headline notebook is a single run, seed 42 |

## Files

- `notebooks/` holds the executed Kaggle notebooks exactly as they ran, outputs
  included. They are the evidence, so please do not re-save them.
- `prudensnet.py` is the architecture from the headline notebook written out
  cleanly, with ParamPad off by default and the real parameter count
  (1,130,382) asserted.

The MRI dataset is public and is not committed here.
