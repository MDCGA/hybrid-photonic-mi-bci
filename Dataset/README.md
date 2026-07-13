# Dataset

Raw EEG datasets are intentionally kept out of git because the BCICIV ASCII
files are large.

Expected local layout for the current experiments:

```text
Dataset/
  BCICIV_1_asc/
    BCICIV_calib_ds1a_cnt.txt
    BCICIV_calib_ds1a_mrk.txt
    BCICIV_calib_ds1a_nfo.txt
    ...
    BCICIV_calib_ds1g_cnt.txt
    BCICIV_calib_ds1g_mrk.txt
    BCICIV_calib_ds1g_nfo.txt
  BNCI2014_004/
    BCICIV_2b_gdf.zip
    gdf/
      B0101T.gdf
      ...
      B0905E.gdf
```

The current FBCSP design comparison expects `Dataset/BCICIV_1_asc` to exist locally:

```bash
python examples/run_fbcsp_design_comparison.py
```

The BNCI2014_004 personalization test expects the BCI Competition IV 2b GDF
files under `Dataset/BNCI2014_004/gdf/`:

```bash
python examples/run_bnci2014_004_personalization.py
```
