# Use the official Python 3.10 slim image as the base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Update system packages and install required dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    gcc \
    libdbus-1-dev \
    libglib2.0-dev \
    libssl-dev \
    pkg-config \
    meson \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first (to leverage Docker caching)
COPY requirements.txt .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip  

# Install torch first with a CPU-only version
RUN pip install --no-cache-dir torch==2.0.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Install the remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Copy the .env file (ensure it's not ignored in .dockerignore)
COPY .env .
# ENV OPENAI_API_KEY=<your_default_key>

# Expose the default Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "application.py", "--server.port=8501", "--server.address=0.0.0.0"]
