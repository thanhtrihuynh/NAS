# dashboard_full V5.18 — Full Test Set Evaluation

Adds asynchronous evaluation of every row in `datasets/splits/test.csv`.

## UI

The `Inference Result` panel now contains:

- `Run Full Test Set · Local Integer`
- `Run Full Test Set · PYNQ FPGA Hybrid`
- progress samples / total
- Accuracy
- Macro F1
- average total latency
- average FPGA weighted latency
- failed sample count
- ETA / elapsed time
- 5×5 confusion matrix
- Stop button
- saved CSV + JSON summary paths

The batch test follows the same `Local Integer / PYNQ FPGA Hybrid`
selector used for single-heartbeat inference.

## Backend

Endpoints:

```text
POST /api/test/run
GET  /api/test/latest
GET  /api/test/jobs/{job_id}
POST /api/test/jobs/{job_id}/cancel
```

The evaluation runs in a background thread, so a long PYNQ test does not
hold one HTTP request open for hours.

Results are saved under:

```text
artifacts/test_runs/
```

## Dataset performance fix

`DatasetService.load_beat()` now reuses the existing WFDB record cache
instead of reopening the `.dat/.hea` record for every heartbeat. This is
important when evaluating thousands of test beats.

## Important

Full-test evaluation is a final evaluation step only. It does not retrain,
run NAS, search mixed precision, or modify the deployment model.

PYNQ Hybrid currently has substantial per-heartbeat software/runtime
overhead, so a complete FPGA test can take much longer than Local Integer.


# V5.19 — Per-candidate NAS checkpoints

Every new NAS candidate now receives its own persistent folder:

```text
artifacts/
└── dashboard_nas_jobs/
    └── <job_id>/
        ├── candidates.json
        ├── best_candidate.json
        ├── candidate_000/
        │   ├── config.json
        │   ├── checkpoint.weights.h5
        │   ├── history.json
        │   └── metrics.json
        ├── candidate_001/
        │   ├── config.json
        │   ├── checkpoint.weights.h5
        │   ├── history.json
        │   └── metrics.json
        └── ...
```

`checkpoint.weights.h5` contains the best validation-loss weights from
that candidate's NAS training. The dashboard now shows `Checkpoint: SAVED`
for new candidates.

This is intentionally a weights checkpoint instead of only a full serialized
Keras model. The architecture is already stored in `config.json`, so later
candidate inference can reconstruct `build_nas_model(architecture)` and load
`checkpoint.weights.h5`.

Existing/old NAS runs are not modified retroactively and may show
`Checkpoint: NOT SAVED`.
