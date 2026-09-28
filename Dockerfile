# 1. Use a lightweight Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies required for C-compilers (like LightFM/NumPy)
RUN apt-get update && apt-get install -y gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your requirements and environment files
COPY requirements.txt .

# 5. We will use pip inside Docker for simplicity, pointing to the exact versions we resolved
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir lightfm==1.17

# 6. Copy the rest of your application code
COPY src/ /app/src/
COPY data/ /app/data/

# 7. Expose the port FastAPI runs on
EXPOSE 8000

# 8. Command to run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]