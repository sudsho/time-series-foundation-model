# time-series-foundation-model

Pretrain a small transformer-based time-series foundation model on synthetic + a few public series, then fine-tune for forecasting on M4 mini subset.

Inspired by PatchTST (2023) and the wave of patch-based TS transformer work this year. Goal is small but reasonable: ~1-2M params, single GPU, single weekend of compute.

## approach

```
[univariate series] -> [patches] -> [linear proj] -> [transformer encoder] -> [head]
```

1. Patch the input series (length L) into N non-overlapping patches of size P.
2. Pretrain with masked patch reconstruction (MPR): randomly mask ~40% of patches, predict the masked values back.
3. For forecasting: drop the recon head, attach a linear forecasting head over the last token, fine-tune.
4. Eval on a small M4 slice (Hourly + Daily) with MASE and sMAPE.

## stack

- PyTorch 2.0
- transformers (utilities only)
- FastAPI + uvicorn for the /forecast endpoint
- MLflow for tracking pretrain and finetune runs
- pytest
- Docker + docker-compose for local serving

## results

Tiny model (~1.1M params, d_model=128, 4 layers) pretrained on 5k synthetic series, finetuned on a small M4 surrogate slice (Hourly + Daily, 200 series each). Numbers below are validation, not the official M4 holdout.

| Subset  | sMAPE | MASE  | naive baseline sMAPE |
|---------|-------|-------|----------------------|
| Hourly  | 12.4  | 0.83  | 18.7                 |
| Daily   | 8.1   | 0.71  | 11.2                 |

Notes:
- Pretrain cuts the finetune sMAPE by ~2-3 points on Hourly vs training the encoder from scratch (same hyperparams).
- Freezing the encoder during finetune is worse than full finetune for this small a model.
- Larger configs (d_model=256, 6 layers) didn't help much on the M4 mini slice and overfit fast.

## setup

```
pip install -r requirements.txt
```

## usage

```
# pretrain on the synthetic mix
python -m src.pretrain --config configs/default.yaml

# finetune on M4 mini (downloads a tiny surrogate if M4 CSVs are missing)
python -m src.finetune --config configs/finetune.yaml

# inference from a saved checkpoint
python -m src.predict --ckpt artifacts/finetune/best.pt --series series.npy --context-length 256
```

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

CI config example lives at `ci/test.yml.example`. Move it under `.github/workflows/` once OAuth has the `workflow` scope.

## deploy notes

See `deploy/NOTES.md` for the rough decision log (Fly.io picked over AWS Fargate for cost on a side project).

## repo layout

```
src/
  data.py        # synthetic generators (sine, ar1, trend, rw, ...)
  patches.py     # patchify / unpatchify / patch embedding
  masking.py     # random patch masking for MPR
  model.py       # encoder + recon/forecast heads
  pretrain.py    # MPR pretraining loop with MLflow
  finetune.py    # forecasting finetune with MASE/sMAPE eval
  m4_mini.py     # M4 subset loader (with synthetic fallback)
  metrics.py     # MASE, sMAPE
  predict.py     # CLI inference entrypoint
  api/main.py    # FastAPI service
configs/
  default.yaml   # pretrain config
  finetune.yaml  # finetune config
tests/           # pytest suite (~25 tests)
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
- [x] CI config
- [ ] swap synthetic M4 surrogate for real M4 CSVs
- [ ] try larger pretrain corpus with a few real series mixed in
