# -----------------------------------------------------
# 1. Base Image 
# -----------------------------------------------------
FROM python:3.10-slim

# -----------------------------------------------------
# 2. Set working directory
# -----------------------------------------------------
WORKDIR /app

# -----------------------------------------------------
# 3. Install system dependencies (PyMuPDF, Docx, etc.)
#    This layer is cached unless THIS BLOCK changes.
# -----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgl1 \
        libglib2.0-0 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------
# 4. Copy ONLY requirements.txt first
#    This allows Docker to cache pip install layer.
# -----------------------------------------------------
COPY requirements.txt .

# -----------------------------------------------------
# 5. Install Python dependencies
#    Cached until requirements.txt changes.
# -----------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------
# 6. Copy model folder *separately* (for caching)
#    If the model folder doesn't change → CACHE HIT
# -----------------------------------------------------
COPY mistral_model/ /app/mistral_model/

# -----------------------------------------------------
# 7. Copy source code last (only this changes frequently)
#    When app.py changes → ONLY this layer rebuilds
# -----------------------------------------------------
COPY . .

# -----------------------------------------------------
# 8. Expose streamlit's default port
# -----------------------------------------------------
EXPOSE 8501

# -----------------------------------------------------
# 9. Streamlit run command
# -----------------------------------------------------
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
