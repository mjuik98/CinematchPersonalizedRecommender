PYTHON ?= python

seed-sample:
	$(PYTHON) scripts/create_synthetic_demo_data.py

download-ml1m:
	$(PYTHON) scripts/download_movielens_1m.py

train-sample:
	$(PYTHON) scripts/train_pipeline.py --dataset-source sample

train-ml1m:
	$(PYTHON) scripts/train_pipeline.py --dataset-source movielens_1m

export-report:
	$(PYTHON) scripts/generate_sample_report.py

run:
	uvicorn app.api:create_app --factory --reload

smoke:
	$(PYTHON) scripts/smoke_test.py

test:
	$(PYTHON) -m unittest discover -s tests -v

clean:
	rm -rf storage/artifacts/* storage/reports/* storage/logs/*

.PHONY: seed-sample download-ml1m train-sample train-ml1m export-report run smoke test clean
