FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
# This service defaults to CPU inference; use the CPU PyTorch wheels instead
# of pulling CUDA/cuDNN/NCCL into a container with no GPU runtime configured.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch torchvision \
    && pip install --no-cache-dir -r requirements.txt
COPY backend/app ./app
COPY backend/alembic.ini ./alembic.ini
COPY backend/alembic ./alembic
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
