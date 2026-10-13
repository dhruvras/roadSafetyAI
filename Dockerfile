# Backend image: FastAPI + the detector
FROM python:3.12-slim-bookworm

# system libraries OpenCV needs (libGL, glib) + git (one requirement is installed from a git URL)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. install the Python packages first. This layer is cached: it only re-runs when requirements.txt changes,
#    not every time you edit a .py file.
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# 2. then copy the code (changes often, so it comes last)
COPY *.py ./
COPY backend ./backend

EXPOSE 8000
# 0.0.0.0 = accept connections from outside the container (127.0.0.1 would only be reachable from inside it)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
