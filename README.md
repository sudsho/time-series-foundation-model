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

## status

WIP. Repo is being built up over October 2023.

## setup

```
pip install -r requirements.txt
```

## todo

- [ ] data generators
- [ ] patch embedding
- [ ] model
- [ ] pretrain loop
- [ ] finetune loop
- [ ] eval (MASE, sMAPE)
- [ ] FastAPI endpoint
- [ ] tests
- [ ] Docker
