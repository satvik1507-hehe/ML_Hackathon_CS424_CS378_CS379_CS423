.PHONY: install api ui inventory

install:
	python -m pip install -e .

api:
	uvicorn pathsnap.api.main:app --app-dir src --host 0.0.0.0 --port 8000

ui:
	streamlit run ui/app.py

inventory:
	python -m pathsnap.data.inventory

