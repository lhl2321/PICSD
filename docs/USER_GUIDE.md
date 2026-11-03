# PICSD user guide

This document describes the inputs, outputs, options and expected behaviour of
the PICSD training pipeline (`PICSD_train.ipynb`). For step-by-step examples, see
[`TUTORIAL.md`](TUTORIAL.md).

## 1. Pipeline overview

1. All well files in `--data_dir` are loaded. Each *slice* (one sample id) must
   contain exactly `--seq_length` depth rows.
2. Wells listed in `--test_wells` form the test set. The remaining wells are
   shuffled with the given seed and split into training and validation wells by
   `--train_ratio`.
3. Consecutive slices of each well are grouped into windows of 5 slices
   (sequence blocks) for CRF training and evaluation.
4. Missing values are imputed with training-set column means, and every channel
   is min-max normalized with training-set statistics.
5. The three-path model produces emission scores per slice. The whole-well CRF
   prior is applied for decoding, using the order supplied in `--stratum_order`.
6. Metrics, confusion matrices, per-well tables, a detailed report and the best
   checkpoint are written to `--save_dir`.

## 2. Input data

### 2.1 Directory layout

`--data_dir` must contain one file per well. CSV (`.csv`) and Excel
(`.xlsx`, `.xls`) files are accepted, and mixed extensions are allowed. The
file name without extension becomes the well name used in reports and in
`--test_wells`.

```text
data/synthetic/
├── SYN_WELL_01.csv
├── ...
├── SYN_WELL_10.csv
├── SYN_TEST_01.csv
└── SYN_TEST_02.csv
```

### 2.2 Required columns

| Role | Option | Default |
| --- | --- | --- |
| Slice identifier | `--sample_id_col` | `sample_id` |
| Slice label | `--label_col` | `formation_label` |
| Model input channels | `--feature_cols` | the 17 columns below |

Default 17 channels (`DEFAULT_FEATURE_COLS`):

| Channel | Meaning |
| --- | --- |
| `GR_INPEFA` | Integrated prediction-error-like trend attribute of `GR` |
| `GR` | Gamma ray |
| `CNL` | Compensated neutron log |
| `AC` | Acoustic transit time |
| `CAL` | Caliper |
| `DEN` | Bulk density |
| `GR_A2`, `GR_D1` | db4 level-2 approximation and level-1 detail of `GR` |
| `DEN_A1` | db4 level-1 approximation of `DEN` |
| `CNL_A3`, `CNL_D3` | db4 level-3 approximation and level-2 detail of `CNL` |
| `AC_A1` | db4 level-1 approximation of `AC` |
| `vector_dim_1` … `vector_dim_5` | Lithology word-vector components |

Pass a comma-separated list to use another channel set:

```bash
--feature_cols "GR,AC,DEN,CNL,vector_dim_1,vector_dim_2,vector_dim_3,vector_dim_4,vector_dim_5"
```

Every listed channel must exist in every well file.

### 2.3 Slice structure and ordering

* One slice = one unique value of `--sample_id_col` with `--seq_length` rows
  (default 120, i.e. 15 m at a 0.125 m depth step).
* Slices are ordered by the integer that follows `_slice_` in the sample id.
  Keep this naming convention (`<well>_slice_<n>`) when preparing new data.
* Slices whose row count differs from `--seq_length` are skipped with a
  message; a well with no valid slice is skipped.

### 2.4 Labels and the stratigraphic prior

Labels are arbitrary strings. They are encoded with `scikit-learn`
`LabelEncoder` (alphabetical order), so the encoded class indices depend on the
label names. The stratigraphic prior is supplied separately as an ordered list
of names, from the shallowest (top) member to the deepest (bottom) member:

```bash
--stratum_order "J5,J4,J3,J2,J1"
```

The list must contain every label class exactly once and no unknown name; the
pipeline stops with an error otherwise. Names are converted to encoded indices
and passed to the CRF, which allows:

* self-transitions (`A -> A`), and
* downward transitions between **adjacent** members of the order.

All other transitions receive `--transition_penalty` (default `-3.0`) as a fixed
soft penalty. Because the penalty is soft, a transition is discouraged rather
than forbidden.

## 3. Command-line options

Options can be passed when the pipeline runs as a script. Inside Jupyter, set
them in the `NOTEBOOK_ARGS` list of the main cell.

### Basic settings

| Option | Default | Description |
| --- | --- | --- |
| `--random_seed` | `0` | Seed for Python, NumPy and PyTorch; also seeds the well split |
| `--data_dir` | `./data/synthetic` | Directory with one well file per well |
| `--seq_length` | `120` | Rows per slice |
| `--train_ratio` | `0.8` | Fraction of non-test wells used for training |
| `--test_wells` | `SYN_TEST_01,SYN_TEST_02` | Comma-separated held-out well names |
| `--stratum_order` | `J5,J4,J3,J2,J1` | Members ordered top (shallow) to bottom (deep) |
| `--sample_id_col` | `sample_id` | Slice identifier column |
| `--label_col` | `formation_label` | Slice label column |
| `--feature_cols` | empty | Comma-separated channels; empty uses the default 17 |

### Model settings

| Option | Default | Description |
| --- | --- | --- |
| `--emb_dim` | `128` | Embedding dimension |
| `--depth` | `2` | Number of SoftShape layers |
| `--sparse_rate` | `0.3` | Final target sparsity rate |
| `--shape_size` | `60` | Shapelet window size |
| `--shape_stride` | `5` | Shapelet stride |
| `--moe_num_experts` | `4` | Number of MoE experts |
| `--moe_top_k` | `2` | Experts selected per token |
| `--transformer_nhead` | `4` | Attention heads |
| `--transformer_layers` | `2` | Transformer encoder layers |

`shape_size` and `shape_stride` must leave at least one patch:
`(seq_length - shape_size) / shape_stride + 1 > 0`.

### Training settings

| Option | Default | Description |
| --- | --- | --- |
| `--warm_up_epoch` | `5` | Epochs without sparsification |
| `--sparse_ramp_epoch` | `20` | Epochs to ramp sparsity from 0 to `--sparse_rate` |
| `--moe_loss_rate` | `0.02` | Weight of the MoE balance loss |
| `--loss` | `cross_entropy` | Auxiliary loss: `cross_entropy` or `focal` |
| `--optimizer` | `adam` | `adam` (AdamW) or `sgd` |
| `--lr` | `0.001` | Initial learning rate |
| `--weight_decay` | `1e-5` | Weight decay |
| `--batch_size` | `32` | Batch size |
| `--epoch` | `200` | Maximum number of epochs |
| `--cuda` | `cuda:0` | Device string; CPU is used if CUDA is unavailable |
| `--aux_loss_rate` | `0.3` | Weight of the auxiliary classification loss |
| `--label_smoothing` | `0.05` | Label smoothing for cross-entropy |
| `--focal_gamma` | `2.0` | Focal-loss gamma |
| `--transition_penalty` | set in code (`-3.0`) | Fixed CRF penalty for invalid transitions |

### Output settings

| Option | Default | Description |
| --- | --- | --- |
| `--save_dir` | `./result_specified_split` | Output directory |
| `--save_csv_name` | experiment-based | Summary CSV base name |
| `--save_model_name` | experiment-based | Checkpoint file name |

The experiment name and the seed-based file names are fixed in the main cell:
`MoE4_Top2_Sparse03` and `seed_<random_seed>`.

## 4. Outputs

Every file is written to `--save_dir`.

| File | Content |
| --- | --- |
| `best_model_MoE4_Top2_Sparse03_seed_<seed>.pth` | `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_val_f1`, `best_val_acc`, `args`, `used_seed`, `class_weights`, `class_order` |
| `log_stratigraphy_MoE4_Top2_Sparse03_seed_<seed>.csv` | One row with overall/macro/micro metrics, training time and configuration |
| `detailed_validation_report_MoE4_Top2_Sparse03_seed_<seed>.txt` | Metric sections, per-well tables, per-slice predictions, sequence-error lists |
| `confusion_matrix_seed_<seed>.csv` / `.png` | Validation confusion matrix (labels as row/column names) |
| `test_confusion_matrix_seed_<seed>.csv` / `.png` | Test confusion matrix |
| `training_history_MoE4_Top2_Sparse03_seed_<seed>.png` | Loss, accuracy, learning rate and configuration panel |
| `well_<well>_validation_result_seed_<seed>.csv` | Per-slice validation predictions for one well |
| `well_<well>_test_result_seed_<seed>.csv` | Per-slice test predictions for one well |

Per-slice columns in the well tables: `sample_id`, `true_label_name`,
`pred_label_name`, `is_correct`. The detailed report additionally lists every
transition that constitutes a sequence error, with well name, previous/next
sample, transition type (`reverse_transition` or `layer_skip`) and label names.

The checkpoint is a PyTorch dictionary; `args` records the options used and
`class_order` records the encoded stratigraphic order, so the architecture and
the CRF prior can be rebuilt exactly.

## 5. Expected behaviour

* **Training** uses 5-slice windows; **validation and test** aggregate the
  emission scores of every slice over all overlapping windows and then run one
  Viterbi decoding per well. Per-well accuracy is therefore based on a whole-well
  sequence, not on independent windows.
* The best checkpoint is selected by validation macro-F1. Model saving and
  early stopping are only active once the MoE warm-up phase has ended
  (`epoch + 1 >= --warm_up_epoch + 1`); during warm-up the early-stop counter is
  reset. With a very small `--epoch`, no checkpoint is written and the final
  evaluation section is skipped. Use `--warm_up_epoch 0` for short test runs.
* Class weights are computed on the training slices and normalized to mean 1.
* The reported loss is `CRF loss + aux_loss_rate * auxiliary loss + moe_loss_rate * MoE loss`.
* On the synthetic dataset, a converged model can reach (near) perfect slice
  accuracy, because several channels are deterministic functions of the
  lithology class. This is intended as a functional test, not as a performance
  claim.
* Changing `--stratum_order` changes both the decoding constraint and the
  sequence-error statistics. An order that contradicts the data will lower
  accuracy and increase the sequence error count; this is expected.

## 6. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `No well files ... found` | `--data_dir` is wrong or contains no `.csv/.xlsx/.xls` files |
| `Missing feature columns` | `--feature_cols` lists a channel absent from a well file |
| `Skip <well>/<slice>: N rows, expected 120` | The slice length differs from `--seq_length`; fix the data or the option |
| `None of the requested test wells were found` | `--test_wells` names do not match file names |
| `At least two non-test wells are required` | Too few wells outside the test set |
| `The names in --stratum_order must match the labels` | Order names must equal the label values exactly, one per class |
| `No module named 'models'` | Start Jupyter from the repository root so `models/PICSD.py` is importable |
| CUDA requested but unavailable | The pipeline automatically falls back to CPU |

## 7. Reproducibility

* `--random_seed` controls Python, NumPy and PyTorch seeds and the well split.
* The split is deterministic for a fixed data directory and seed.
* The synthetic dataset is fixed on disk and does not depend on network access.
* Results may differ slightly across PyTorch versions, devices and CPU/GPU
  implementations of convolutions and matrix multiplications.
