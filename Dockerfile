# Dockerfile (fixed)
FROM python:3.10-slim

# system deps - use libgl1 (available) instead of libgl1-mesa-glx
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# create uploads folder
RUN mkdir -p /app/uploads

EXPOSE 5000

# run using gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app", "--workers", "1", "--threads", "4"]

