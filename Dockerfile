# Dockerfile
# Builds the Sociogram Generator as a container for deployment to
# Azure App Service (Australia East).

FROM python:3.11-slim

# Keeps Python from buffering stdout/stderr (important for Azure log streaming)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (separate layer — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app_sociogram.py auth.py audit.py ./

# Azure App Service passes the port via the WEBSITES_PORT env var.
# Streamlit's default is 8501; we expose that and let the startup command bind to it.
EXPOSE 8501

# Streamlit config: disable the "are you sure you want to navigate away" dialog,
# turn off the usage stats ping, and bind to all interfaces so Azure can reach it.
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app_sociogram.py", "--server.port=8501", "--server.address=0.0.0.0"]
