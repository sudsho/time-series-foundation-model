.PHONY: install test lint pretrain finetune predict serve clean

install:
	pip install -r requirements.txt

test:
	pytest -q tests/

lint:
	python -m py_compile src/*.py

pretrain:
	python -m src.pretrain --config configs/default.yaml

finetune:
	python -m src.finetune --config configs/finetune.yaml

predict:
	python -m src.predict --ckpt artifacts/finetune/best.pt

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf artifacts/ checkpoints/ mlruns/ .pytest_cache/ __pycache__/
