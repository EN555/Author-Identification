fmt:
	isort -l 79 --profile black src product/backend tests
	black -l 79 src product/backend tests

lint:
	isort -l 79 --profile black -c src product/backend tests
	black -l 79 --check src product/backend tests
	flake8 src product/backend tests --max-line-length 95 --ignore=E203,E266,W503,E402
	mypy src product/backend tests --ignore-missing-imports

install: 
	pip install -r requirments.txt

pull_data:
	poetry run dvc pull

test:
	pytest tests/unit/ -maxfail=1 --full-trace --cov-report term-missing:skip-covered --cov-fail-under=10 --cov src

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache

build:
	docker build -t yd5463/author-identifier:frontend ./product/frontend
	docker build -t yd5463/author-identifier:backend -f ./product/backend/Dockerfile .
deploy:
	docker push yd5463/author-identifier:frontend
	docker push yd5463/author-identifier:backend
	helm template ./helm-charts/service/ --set image.tag="frontend" > out.yaml
	kubectl apply -f out.yaml
	helm template ./helm-charts/service/ --set image.tag="backend" > out.yaml
	kubectl apply -f out.yaml
