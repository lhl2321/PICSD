# Synthetic well-log dataset

This folder contains a fully synthetic dataset that reproduces the table layout
of the field data used in the manuscript. The original wells are proprietary
and cannot be redistributed; this dataset allows reviewers and users to execute
the complete PICSD pipeline without any confidential material.

## Files

12 well files, one per well, in CSV format:

| File | Role |
| --- | --- |
| `SYN_WELL_01.csv` … `SYN_WELL_10.csv` | candidate training / validation wells |
| `SYN_TEST_01.csv`, `SYN_TEST_02.csv` | held-out test wells |

Because the split is performed at well level, the two test wells are never seen
during training.

Each well has 3600 rows = 30 slices × 120 depth samples, with a uniform depth
step of 0.125 m (375 m of section per well).

## Columns

| Column | Meaning |
| --- | --- |
| `Depth` | Measured depth (m), step 0.125 m |
| `GR` | Gamma ray (API) |
| `AC` | Acoustic transit time (µs/m) |
| `CAL` | Caliper (cm) |
| `CNL` | Compensated neutron log (%) |
| `DEN` | Bulk density (g/cm³) |
| `Lithology` | Descriptive lithology of the slice |
| `Formation` | Stratigraphic member name |
| `sample_id` | Unique slice identifier, `<well>_slice_<n>` |
| `formation_label` | Slice label (`J1` … `J5`) |
| `GR_INPEFA` | Integrated prediction-error-like trend attribute of `GR`, scaled to about [-1, 1] |
| `vector_dim_1` … `vector_dim_5` | 5-D lithology word vector: the numerical representation of the lithology description of the slice (constant inside a slice) |
| `GR_A1` … `GR_A8`, `GR_D1` … `GR_D8` | db4 8-level wavelet decomposition of `GR`: `A<k>` is the level-k approximation coefficient and `D<k>` the level-k detail coefficient |
| `AC_A1` … `AC_A8`, `AC_D1` … `AC_D8` | the same decomposition of `AC` |
| `CAL_A1` … `CAL_A8`, `CAL_D1` … `CAL_D8` | the same decomposition of `CAL` |
| `CNL_A1` … `CNL_A8`, `CNL_D1` … `CNL_D8` | the same decomposition of `CNL` |
| `DEN_A1` … `DEN_A8`, `DEN_D1` … `DEN_D8` | the same decomposition of `DEN` |

A well file therefore has 96 columns: 11 base/metadata columns, 5 word-vector
components and 5 logs × 8 levels × 2 coefficient types = 80 wavelet columns.

The wavelet transform is a pyramid-style discrete wavelet transform with the
Daubechies-4 (`db4`) wavelet and `symmetric` boundary handling. It is computed
on the whole well curve (3600 depth samples, which supports 8 levels), and the
approximation and detail coefficients of every level are interpolated back onto
the depth samples, so each row carries the complete 8-level description of its
depth position. The coefficients are not normalised and therefore scale with
`sqrt(2)` per level (`A8` has a much larger magnitude than `A1`); this is the
standard PyWavelets convention.

The 17 channels used by the model are listed in `PICSD_train.ipynb`
(`DEFAULT_FEATURE_COLS`): `GR_INPEFA`, the five raw logs, six wavelet
coefficients (`GR_A2`, `GR_D1`, `DEN_A1`, `CNL_A3`, `CNL_D3`, `AC_A1`) and the
five word-vector components. The remaining columns are metadata, descriptive
attributes or wavelet coefficients that the default configuration does not use.
They can be selected with the `--feature_cols` option of the training pipeline.

## Labels and stratigraphic order

| Label | Member | Approximate position |
| --- | --- | --- |
| `J5` | upper member | shallowest |
| `J4` | — | |
| `J3` | — | |
| `J2` | — | |
| `J1` | lower member | deepest |

Labels follow the normal order from top to bottom: `J5 → J4 → J3 → J2 → J1`.
This order is passed to the training pipeline with
`--stratum_order "J5,J4,J3,J2,J1"` (the default). It defines the CRF transition
constraint built by `models/PICSD.py` (`StratigraphyCRFNet.class_order`) and the
transition ranking used by `count_sequence_errors` in
`PICSD_train.ipynb`.

Any other order can be supplied as long as the names match the label values
exactly, e.g. `--stratum_order "J2,J3,J4,J5,J1"`; the CRF mask and the
sequence-error statistics are both derived from the order.

## Using the dataset with the pipeline

The default settings of `PICSD_train.ipynb` already point at this folder:

| Option | Value |
| --- | --- |
| `--data_dir` | `./data/synthetic` |
| `--test_wells` | `SYN_TEST_01,SYN_TEST_02` |
| `--sample_id_col` | `sample_id` |
| `--label_col` | `formation_label` |
| `--stratum_order` | `J5,J4,J3,J2,J1` |
| `--seq_length` | `120` |

## How the data were generated

The dataset was produced with a deterministic synthetic generator that draws
class-conditional log responses (`CLASS_PROFILES`), adds a slow vertical drift,
a random walk component and white noise, then attaches a fixed word vector to
every lithology term (`LITHOLOGY_EMBEDDINGS`), computes `GR_INPEFA` and derives
the db4 8-level wavelet coefficients of all five logs. Slice labels are
drawn from a monotone top-to-bottom sequence with random thicknesses; wells may
start in `J5` or `J4` and end in `J1` or `J2`, while the test wells always cover
all five members.

The generator is not part of this repository, so the shipped CSV files are the
reference dataset: every training and evaluation step that uses this synthetic
data can be reproduced directly from these files with a fixed random seed.

## Important limitations

* The data are synthetic and are **not** a substitute for field measurements.
  Metrics obtained on this dataset must not be compared with the accuracy
  reported in the manuscript.
* The class-conditional log responses partially overlap, but the word-vector
  and wavelet columns are deterministic functions of the lithology class, so
  the task is considerably easier than real field interpretation. A converged
  model can reach (near) perfect accuracy on this dataset; the CRF constraint
  removes most sequence errors but does not guarantee a fully
  correct segmentation on noisier data.
* Any resemblance between the generated values and a specific real well is
  coincidental.

## License

The synthetic dataset is released under the MIT License of this repository.
