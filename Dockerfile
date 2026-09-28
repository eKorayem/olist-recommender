# 1. Use a lightweight Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies required for C-compilers (like LightFM/NumPy)
RUN apt-get update && apt-get install -y gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your requirements and environment files
COPY requirements.txt .

# 5. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir lightfm==1.17

# 6. Copy application code and data
COPY src/ /app/src/
COPY data/ /app/data/

# 7. Expose ports for both FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# 8. Default command (for backend)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]