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

## Quick start (runs offline)

No downloads, no GPU, no API keys, no external services. The smoke builds a tiny
deterministic synthetic dataset (seasonal + trend), pretrains a tiny encoder with
masked patch reconstruction, finetunes a forecasting head, and forecasts a held-out
series. It exercises the same code paths as the full pipeline (patchify, patch
embedding, encoder, recon head, forecast head, predict, metrics).

```
python scripts/smoke.py      # or: make smoke
```

Real output (a couple of seconds on a laptop CPU):

```
==============================================================
time-series-foundation-model :: offline tiny-CPU smoke
==============================================================
device=cpu  context_length=64  patch_len=8  n_patches=8  horizon=8

[1] synthetic seasonal+trend dataset: (96, 72) (train=88, held=8)
    tiny model: d_model=32 n_layers=2 n_heads=4 -> 19,752 params

[2] patchify  (88, 64) -> (88, 8, 8)  (B,N,P)  OK
    embed     (88, 8, 8) -> (88, 8, 32)  (B,N,D)  OK

[3] MPR pretrain (masked-patch recon): MSE 1.2796 -> 0.7709

[4] forecast finetune (MSE on 8-step horizon): 1.4996 -> 0.0028

[5] forecast on a held-out series (predicted vs actual):
    idx |  actual  | predicted
      0 |  -0.4372 |  -0.4385
      4 |   0.4557 |   0.5036
      7 |  -0.3346 |  -0.1634
    sMAPE=42.414%   MASE=0.272

[6] mean sMAPE over 8 held-out series: 28.980%

==============================================================
SMOKE PASSED: pipeline runs offline, losses decrease, shapes OK
==============================================================
```

MASE < 1 means the finetuned forecaster beats the naive one-step baseline on the
held-out series. The smoke asserts the reconstruction MSE and forecasting MSE both
decrease and that the patch-embedding and forecast output shapes are correct.

Run the tests:

```
pytest -q tests/     # 38 passed
```

### What the full model / GPU / real data add

The smoke is deliberately tiny so it runs anywhere. The shipped configs describe the
real setup, which is heavier and benefits from a GPU:

- **Bigger model**: `configs/default.yaml` uses `d_model=128, n_layers=4` (about 0.54M
  params) versus the smoke's `d_model=32, n_layers=2` (about 20K params).
- **Real pretraining**: `python -m src.pretrain --config configs/default.yaml` builds a
  5000-series synthetic mix, masks 40% of patches, and trains for 20 epochs with MLflow
  tracking (the headline masked-patch-reconstruction run).
- **Forecasting finetune**: `python -m src.finetune --config configs/finetune.yaml`
  attaches the forecast head and evaluates sMAPE / MASE. It uses a synthetic M4-style
  surrogate unless `--data-dir` points at real M4 CSVs (nothing is downloaded).
- **Zero-shot / GPU**: larger context windows, more layers, and the full synthetic mix
  are where a GPU matters; the smoke covers the same math on CPU in seconds.

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
