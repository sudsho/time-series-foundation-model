# deploy notes

## local

```
docker compose up --build
# api at http://localhost:8000
# /docs for swagger
```

If you want MLflow too:
```
docker compose --profile dev up
# mlflow at http://localhost:5000
```

## hosted (options considered)

Two hosting options considered for a future live demo:

1. **AWS ECS Fargate** for the API. Push the image to ECR, point a Fargate
   service at it, ALB in front. Works, but the ALB alone is roughly $25/month
   idle on the smallest tier, not worth it for a portfolio piece.

2. **Render.com or Fly.io** with the same Dockerfile. Cheaper for a tiny side
   project; cold-start latency is longer because the model has to load on boot.

Nothing is deployed from this repo yet. No `fly.toml`, no Terraform, no
`render.yaml`. The image builds locally and runs via `docker compose`.

The mlflow service is `dev` profile only, never intended for public exposure
because the default MLflow UI has no auth.

## checkpoints

The image ships *without* model weights. Mount a volume with
`/app/artifacts/finetune/best.pt` or rebuild the image with the file copied in.

## env vars

| name        | default                              | meaning            |
|-------------|--------------------------------------|--------------------|
| MODEL_CKPT  | /app/artifacts/finetune/best.pt      | finetuned weights  |
