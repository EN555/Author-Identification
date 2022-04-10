.PHONY: notebook docs
.EXPORT_ALL_VARIABLES:

install: 
	@echo "Installing..."
	pip install -r requirments.txt

activate:
	@echo "Activating virtual environment"
	source /venv/Scripts/activate

pull_data:
	poetry run dvc pull

test:
	pytest

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache