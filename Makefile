ENV := KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error

.PHONY: install data eda train fast app clean

install:
	pip install -r requirements.txt

data:
	$(ENV) python -m scripts.make_dataset

eda:
	$(ENV) python -m scripts.eda

train:
	$(ENV) python setup.py

fast:
	$(ENV) FAST=1 python setup.py

app:
	$(ENV) python main.py

clean:
	rm -rf __pycache__ scripts/__pycache__ data/processed/* data/outputs/plots/* models/_smoke
