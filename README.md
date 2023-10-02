# time-series-foundation-model

Pretrain a small transformer-based time-series foundation model on synthetic + a few public series, then fine-tune for forecasting on M4 mini subset.

WIP. Inspired by PatchTST / TimesNet style patch-based modeling that was getting traction this year.

## plan

- patch the input series, project to embedding
- pretrain with masked patch reconstruction (BERT style but on patches)
- swap a forecasting head, fine-tune
- eval with MASE / sMAPE on a small M4 slice
- expose a /forecast endpoint via FastAPI

## stack

PyTorch 2.0, transformers (HF) for utilities, FastAPI for serving.

still drafting; pieces will land over the next couple weeks.
