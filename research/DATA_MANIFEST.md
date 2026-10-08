# Data manifest

All model inputs come from the frozen files in `data/raw/`. Scripts must not
download data when a frozen file exists. Yahoo Finance revises history without
notice, so a fresh download is not reproducible.

| File | Rows | Range | Downloaded | SHA-256 |
|---|---:|---|---|---|
| `wti_CL=F_2010-01-04_to_2024-12-30_downloaded_2026-08-11.csv` | 3,771 | 4 Jan 2010 to 30 Dec 2024 | 11 Aug 2026 | `23e008de49e9bfa89394ecdb9d7018ce5fcc43ef7dc34753a43c14c0c2175166` |
| `wti_CL=F_holdout_2025-01-02_to_2026-08-31_downloaded_2026-09-01.csv` | 418 | 2 Jan 2025 to 31 Aug 2026 | 1 Sep 2026 | `9d1d63aa09b37ca357671b7411ef81db206603c84e9159f7b2755c815e95f4b5` |

Source: Yahoo Finance, ticker `CL=F`, `auto_adjust=True`, `Close` column.
The adjustment flag has no effect on a futures series. It is recorded for
completeness.

## Estimation sample

The 2010 to 2024 file is the estimation sample. Its prices are identical to
the `price` column of `outputs/data_audit/wti_audit_data.csv` produced on
11 August 2026, so every result recorded before this manifest remains valid.

## Sealed holdout

The 2025 to 2026 file is a sealed out-of-sample holdout. No script reads it
before Stage 6. No model choice may be revised after it is opened. See
`GATES.md`, gate G7.

## Verification

```bash
sha256sum data/raw/*.csv
```
