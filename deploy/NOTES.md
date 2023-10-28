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

## hosted (rough plan)

Tried two options while building this:

1. **AWS ECS Fargate** for the API. Push the image to ECR, point a Fargate
   service at it, ALB in front. Works fine but ~$25/month idle just for the
   ALB on the smallest tier , not worth it for a portfolio piece that mostly
   sits idle.

2. **Render.com or Fly.io** with the same Dockerfile. Way cheaper for a tiny
   side project; cold-start latency isn't great (~10-15s) since the model has
   to load on boot, but acceptable.

Final pick for the live demo: Fly.io with `fly launch`, 1 shared-cpu instance,
1GB RAM. Can add more later.

The mlflow service is `dev` profile only , never deployed publicly because the
default mlflow UI has no auth.

## checkpoints

The image ships *without* model weights. Mount a volume with
`/app/artifacts/finetune/best.pt` or rebuild the image with the file copied in.

## env vars

| name        | default                              | meaning            |
|-------------|--------------------------------------|--------------------|
| MODEL_CKPT  | /app/artifacts/finetune/best.pt      | finetuned weights  |
