.PHONY: install test run api pipeline docker docker-test clean help

PYTHON := python3
PIP := pip3

help:
	@echo "Commands: install, test, run, api, pipeline, docker, docker-test, clean"

install:
	$(PIP) install -r requirements.txt

test:
	pytest test_pipeline.py -v --tb=short

run:
	streamlit run app.py --server.port=8501 --server.address=localhost

api:
	uvicorn api:app --host 0.0.0.0 --port 8000 --reload

pipeline:
	$(PYTHON) pipeline.py

docker:
	docker-compose up --build -d

docker-test:
	docker-compose run --rm tests

clean:
	rm -f pipeline.log *.db *.csv
	rm -rf __pycache__ .pytest_cache *.pyc
