# PICSD tutorial

Five typical use cases. Each section assumes the working directory is the
repository root and the environment from [`README.md`](../README.md) is active.

## 1. Run the demo on the synthetic dataset

1. Start JupyterLab:

   ```bash
   jupyter lab PICSD_train.ipynb
   ```

2. Run all cells. The defaults are:

   * data: `./data/synthetic`
   * test wells: `SYN_TEST_01`, `SYN_TEST_02`
   * stratigraphic order: `J5 -> J4 -> J3 -> J2 -> J1`
   * 200 epochs

3. Inspect `./result_specified_split/`. The summary CSV and the detailed text
   report contain validation and test metrics; the PNG files show confusion
   matrices and training curves.

The synthetic data are intentionally easy. A converged run can reach (near)
perfect accuracy; this confirms that the pipeline works, and is not a
performance result.

## 2. Run headlessly with command-line options

Convert the notebook to a script once, then pass options:

```bash
jupyter nbconvert --to script PICSD_train.ipynb
python PICSD_train.py --epoch 20 --save_dir ./results_demo
```

A short functional check that also produces a checkpoint and the final
evaluation files:

```bash
python PICSD_train.py --epoch 3 --warm_up_epoch 0 --sparse_ramp_epoch 1 --cuda cpu
```

`--warm_up_epoch 0` is needed for very short runs because checkpoint saving
starts after the MoE warm-up.

## 3. Use your own stratigraphic order

Suppose the members are named `A`, `B`, `C` and are encountered from top to
bottom in that order. Pass the names exactly as they appear in the label
column:

```bash
python PICSD_train.py --stratum_order "A,B,C"
```

The pipeline:

1. encodes the labels found in the data,
2. converts the names to encoded indices,
3. builds the CRF mask so that `A -> B` and `B -> C` are valid, and
4. counts sequence errors against the same order.

The list must cover every class exactly once, in the correct vertical order.
To change the order inside Jupyter, edit the `NOTEBOOK_ARGS` list in the main
cell:

```python
NOTEBOOK_ARGS = ['--stratum_order', 'A,B,C', '--epoch', '50']
```

Reversing the order by mistake is a quick way to see the effect of the prior:
validation accuracy drops and the report lists many `reverse_transition`
sequence errors.

## 4. Train on your own well files

1. Put one file per well in a single directory, for example `my_data/`.
   CSV and Excel are both accepted.
2. Make sure each file contains:

   * a slice identifier column (`--sample_id_col`, default `sample_id`) with
     values ending in `_slice_<integer>`,
   * a label column (`--label_col`, default `formation_label`),
   * every channel listed in `--feature_cols`.

3. Run with the column names and order that match your data:

   ```bash
   python PICSD_train.py \
       --data_dir my_data \
       --test_wells "Well_07,Well_08" \
       --stratum_order "Top,Upper,Middle,Lower,Bottom" \
       --sample_id_col sample_id \
       --label_col formation_label \
       --feature_cols "GR,AC,DEN,CNL,CAL"
   ```

If your slices are not 120 rows, set `--seq_length` accordingly. If your slice
ids do not follow the `_slice_<n>` convention, the loader still groups by the
identifier but the numeric sort falls back to `0` for every slice, which
destroys the vertical order; rename the ids or adjust
`extract_slice_number` in the notebook.

## 5. Read and use the outputs

| Question | File |
| --- | --- |
| How good is the model overall? | `log_stratigraphy_*.csv` |
| Which classes are confused? | `confusion_matrix_seed_*.csv` / `.png` |
| Which slices are wrong in a well? | `well_<well>_*_result_seed_*.csv` |
| Where are the sequence errors? | `detailed_validation_report_*.txt`, section 10/11 |
| Can I reload the model? | `best_model_*.pth` |

Reloading a checkpoint requires rebuilding the same architecture, because the
checkpoint stores weights and arguments but not Python classes:

```python
import torch
from models.PICSD import SoftShapeTransformerNet, StratigraphyCRFNet

ckpt = torch.load("result_specified_split/best_model_MoE4_Top2_Sparse03_seed_0.pth",
                  map_location="cpu")
saved = ckpt["args"]

base = SoftShapeTransformerNet(
    seq_len=saved["seq_len"],
    shape_size=saved["shape_size"],
    num_channels=saved["num_channels"],
    emb_dim=saved["emb_dim"],
    sparse_rate=saved["sparse_rate"],
    depth=saved["depth"],
    num_experts=saved["moe_num_experts"],
    num_classes=saved["num_class"],
    stride=saved["shape_stride"],
    transformer_nhead=saved["transformer_nhead"],
    transformer_layers=saved["transformer_layers"],
    moe_top_k=saved["moe_top_k"],
    sparse_ramp_epoch=saved["sparse_ramp_epoch"],
)
model = StratigraphyCRFNet(
    base_model=base,
    num_classes=saved["num_class"],
    transition_penalty=saved["transition_penalty"],
    class_order=ckpt["class_order"],
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

The checkpoint stores `class_order`, so the CRF prior is restored exactly as it
was used during training. Keep the matching label encoder (label-to-index
mapping) alongside the checkpoint when the model is deployed on new data.

## 6. Run several seeds

Use a shell loop around the exported script:

```bash
for seed in 0 1 2; do
    python PICSD_train.py --random_seed $seed --save_dir ./results_seed_$seed
done
```

Each run appends a row to its own summary CSV, and per-seed files never collide.
