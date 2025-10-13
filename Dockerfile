FROM python:3.11-slim

WORKDIR /app

# Install system deps for pandas/scikit-learn
RUN apt-get update && apt-get install -y build-essential gfortran libatlas-base-dev --no-install-recommends && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install -r /app/requirements.txt

COPY . /app

EXPOSE 8000

CMD ["gunicorn", "main:app", "-b", "0.0.0.0:8000", "--workers", "2"]