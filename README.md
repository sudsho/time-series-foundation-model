# time-series-foundation-model

Small patch-based transformer for univariate time-series pretraining and forecasting, in PyTorch. Pretrain with masked patch reconstruction on a synthetic mix, then attach a forecasting head and fine-tune.

Inspired by PatchTST (2023) and the wave of patch-based TS transformer work this year.

## approach

```
[univariate series] -> [patches] -> [linear proj] -> [transformer encoder] -> [head]
```

1. Patch the input series (length L) into N non-overlapping patches of size P.
2. Pretrain with masked patch reconstruction (MPR): randomly mask ~40% of patches, predict the masked values back (MSE on masked positions only).
3. For forecasting: drop the recon head, attach a linear forecasting head over the flattened token sequence, fine-tune.
4. Eval sMAPE and MASE on a held-out split of the fine-tune series.

## stack

- PyTorch 2.0
- FastAPI + uvicorn for the /forecast endpoint
- MLflow for tracking pretrain runs
- pytest
- Docker + docker-compose for local serving

## model size

With the shipped config (patch_len=16, d_model=128, n_heads=4, n_layers=4, ffn_dim=256), the base encoder is about 0.54M parameters and the forecasting head brings the total to about 0.59M.

## data

All data used by the shipped code is synthetic. `src/data.py` provides a handful of generators (sine, trend, AR(1), seasonal-trend, random walk) and `make_synthetic_dataset` mixes them for pretraining.

`src/m4_mini.py` will read local M4-format CSVs if `--data-dir` is passed and the file exists, otherwise it generates a synthetic surrogate at the requested length. No CSVs are downloaded or committed.

## setup

```
pip install -r requirements.txt
```

## usage

```
# pretrain on the synthetic mix
python -m src.pretrain --config configs/default.yaml

# finetune (uses synthetic M4-style surrogate unless --data-dir points to real CSVs)
python -m src.finetune --config configs/finetune.yaml

# inference from a saved checkpoint
python -m src.predict --ckpt artifacts/finetune/best.pt --series series.npy --context-length 256
```

Notes on the shipped finetune config: only the Hourly subset produces series long enough for the default context_length=256 + horizon=24 filter. The Daily surrogate is length 200 and would be dropped; if you leave both subsets in the config, remove Daily first or shorten context_length.

## serving

```
# spin up the FastAPI service in docker
docker compose up --build
# api at http://localhost:8000  (swagger at /docs)
```

POST a forecast request:
```
curl -X POST http://localhost:8000/forecast \
  -H 'content-type: application/json' \
  -d '{"series": [/* 256 floats */], "context_length": 256}'
```

`/health` returns 200 with `{"ok": true, "model_loaded": <bool>}`. If no checkpoint is mounted, `/forecast` will return 503.

## tests

```
pytest -q tests/
```

38 test functions across `tests/test_api.py`, `test_data.py`, `test_metrics.py`, `test_model.py`, `test_patches.py`.

A CI workflow example lives at `ci/test.yml.example`.

## repo layout

```
src/
  data.py        # synthetic generators (sine, ar1, trend, rw, ...)
  patches.py     # patchify / unpatchify / patch embedding
  masking.py     # random patch masking for MPR
  model.py       # encoder + recon/forecast heads
  pretrain.py    # MPR pretraining loop with MLflow
  finetune.py    # forecasting finetune with MASE/sMAPE eval
  m4_mini.py     # M4-format loader with synthetic fallback
  metrics.py     # MASE, sMAPE
  predict.py     # CLI inference entrypoint
  api/main.py    # FastAPI service
configs/
  default.yaml   # pretrain config
  finetune.yaml  # finetune config
tests/           # pytest suite
ci/              # CI config example
deploy/          # deploy notes
Dockerfile
docker-compose.yml
```

## todo

- [x] data generators
- [x] patch embedding
- [x] model
- [x] pretrain loop
- [x] finetune loop
- [x] eval (MASE, sMAPE)
- [x] tests
- [x] FastAPI endpoint
- [x] Docker
- [ ] swap synthetic surrogate for real M4 CSVs
- [ ] carry per-subset labels through the finetune loop for per-subset metrics
