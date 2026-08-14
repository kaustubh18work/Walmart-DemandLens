install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

run:
	MPLBACKEND=Agg streamlit run streamlit_app.py

test:
	pytest -q

lint:
	ruff check .

compile:
	python -m compileall -q .

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
