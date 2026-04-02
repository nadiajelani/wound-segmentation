FROM tensorflow/tensorflow:2.16.1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --ignore-installed \
    flask==3.0.3 \
    flask-cors==4.0.1 \
    gunicorn==21.2.0 \
    keras==3.3.3 \
    numpy==1.26.4 \
    opencv-python-headless==4.10.0.84 \
    pillow==10.4.0 \
    h5py==3.11.0 \
    protobuf==4.25.3

COPY . .

ENV KERAS_BACKEND=tensorflow
ENV PYTHONUNBUFFERED=1
ENV TF_NUM_INTRAOP_THREADS=2
ENV TF_NUM_INTEROP_THREADS=2
ENV PORT=8080

CMD exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 600 --log-level info
