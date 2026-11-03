# PICSD

PICSD is the reference implementation of a three-path hybrid neural network for
well-log stratigraphic classification. The model combines:

* a **SoftShape mixture-of-experts (MoE)** path with progressive sparsification,
* a **class-specific Transformer** path driven by shapelet-similarity features,
* a **generic Transformer** path over the raw log curves,

and applies a **whole-well conditional random field (CRF) prior** during
decoding. The prior is defined by a user-supplied stratigraphic order: only
self-transitions and downward transitions between adjacent members of that
order are considered valid, and every other transition receives a fixed soft
penalty. The same order is used to count sequence errors in the predictions.

The repository is self-contained. Because the field data used in the manuscript
are proprietary, a **synthetic well-log dataset** is included so that the
complete pipeline (loading, training, whole-well decoding, evaluation and
reporting) can be executed by reviewers and users without confidential
material.

## Repository layout

| Path | Description |
| --- | --- |
| `PICSD_train.ipynb` | Complete training, evaluation and reporting pipeline |
| `models/PICSD.py` | Model classes: `SoftShapeTransformerNet`, `StratigraphyCRFNet` |
| `data/synthetic/` | Synthetic well-log dataset: 12 wells in CSV format |
| `data/README.md` | Dataset description: columns, labels, generation, limitations |
| `docs/USER_GUIDE.md` | Inputs, outputs, options and expected behaviour |
| `docs/TUTORIAL.md` | Step-by-step how-to for typical use cases |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT license |

## Data availability

The original field data cannot be redistributed. `data/synthetic/` reproduces
the table layout of the field data and supports every step of the pipeline.
It contains 10 candidate wells and 2 held-out test wells, all fully synthetic.
See [`data/README.md`](data/README.md) for the column definitions.

The default configuration uses:

* `--data_dir ./data/synthetic`
* `--test_wells SYN_TEST_01,SYN_TEST_02`
* `--stratum_order J5,J4,J3,J2,J1` (shallowest to deepest member)
* 17 feature channels (`DEFAULT_FEATURE_COLS` in `PICSD_train.ipynb`)

## Requirements and computational needs

* Python 3.9 or newer (tested with Python 3.9.25)
* PyTorch 2.x (tested with 2.8.0), `timm`, `pytorch-crf`
* NumPy, pandas, scikit-learn, matplotlib, openpyxl
* JupyterLab or Jupyter Notebook to run the `.ipynb`

All dependencies are listed in [`requirements.txt`](requirements.txt). The
default synthetic experiment is small (12 wells, 3600 rows per well) and runs on
a CPU; a CUDA device is used automatically when available. No special hardware,
external services or pre-trained weights are required.

### Installation

```bash
conda create -n picsd python=3.9 -y
conda activate picsd
pip install -r requirements.txt
```

or, with `venv`:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Quick start

```bash
jupyter lab PICSD_train.ipynb
```

Run all cells. With the default settings the notebook trains on
`data/synthetic`, keeps `SYN_TEST_01` and `SYN_TEST_02` out of training, and
writes models, metrics, confusion matrices and per-well tables to
`./result_specified_split/`.

To run without Jupyter, export the notebook to a script and pass command-line
options:

```bash
jupyter nbconvert --to script PICSD_train.ipynb
python PICSD_train.py --epoch 20 --stratum_order "J5,J4,J3,J2,J1"
```

When the notebook runs inside Jupyter, command-line options can be supplied
through the `NOTEBOOK_ARGS` list at the start of the main cell.

## Defining the stratigraphic prior

`--stratum_order` accepts a comma-separated list of member names ordered from
the shallowest (top) to the deepest (bottom) member. The names must match the
label values that occur in the label column exactly, and every class must
appear exactly once. For example:

```bash
python PICSD_train.py --stratum_order "J5,J4,J3,J2,J1"
python PICSD_train.py --stratum_order "Member A,Member B,Member C"
```

The notebook converts names to encoded class indices and passes them to
`StratigraphyCRFNet(class_order=...)`, which builds the CRF transition mask. The
sequence-error counter uses the same order, so changing the order changes both the
decoding constraint and the reported sequence-error statistics.

## Outputs

All outputs are written to `--save_dir` (default `./result_specified_split/`):

| File | Content |
| --- | --- |
| `best_model_<experiment>_seed_<seed>.pth` | Best checkpoint (state dict, optimizer state, arguments, class weights) |
| `log_stratigraphy_<experiment>_seed_<seed>.csv` | Summary metrics and training configuration |
| `detailed_validation_report_<experiment>_seed_<seed>.txt` | Full report with per-slice predictions and sequence-error lists |
| `confusion_matrix_seed_<seed>.csv` / `.png` | Validation confusion matrix |
| `test_confusion_matrix_seed_<seed>.csv` / `.png` | Test confusion matrix |
| `training_history_<experiment>_seed_<seed>.png` | Loss, accuracy and learning-rate curves |
| `well_<well>_validation_result_seed_<seed>.csv` | Per-slice validation results for one well |
| `well_<well>_test_result_seed_<seed>.csv` | Per-slice test results for one well |

See [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) for the complete option list and
the meaning of every column.

## Reproducibility

* The random seed is fixed with `--random_seed` (default `0`), and the
  train/validation/test split is derived deterministically from it.
* The data are loaded from files, so no network access is required.
* The synthetic dataset is designed as a smoke test, not as a benchmark. A
  converged model can reach (near) perfect accuracy on it; metrics obtained
  there must not be compared with the accuracy reported for the field data in
  the manuscript.

## Documentation

* [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) — inputs, outputs, options, expected behaviour
* [`docs/TUTORIAL.md`](docs/TUTORIAL.md) — typical use cases step by step
* [`data/README.md`](data/README.md) — dataset format and limitations

## License

This repository, including the synthetic dataset, is released under the MIT
License. See [`LICENSE`](LICENSE).

## Citation

If you use this code or the synthetic dataset, please cite the associated
manuscript. Citation metadata will be updated here once the article is
published.
