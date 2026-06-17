FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PORT=7860 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/home/user/.cache/huggingface \
    NLTK_DATA=/home/user/nltk_data \
    DEEP_MODEL=HanfuZhao781/car-review-sentiment-distilbert

WORKDIR /app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY --chown=user . /app
USER user

RUN python -c "import nltk; [nltk.download(p, download_dir='/home/user/nltk_data') for p in ('punkt','punkt_tab','wordnet','omw-1.4')]"
RUN python -c "import os; from transformers import AutoModelForSequenceClassification, AutoTokenizer; m=os.environ['DEEP_MODEL']; AutoTokenizer.from_pretrained(m); AutoModelForSequenceClassification.from_pretrained(m)"

EXPOSE 7860
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:7860", "--timeout", "120", "main:app"]
