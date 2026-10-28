FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY dfs ./dfs
ENV DATA_DIR=/data
EXPOSE 8000
CMD ["uvicorn", "dfs.node:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
