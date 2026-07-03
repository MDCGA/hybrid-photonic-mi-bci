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
```

The default replay command expects `Dataset/BCICIV_1_asc` to exist locally:

```bash
python examples/run_bciciv_replay.py
```
