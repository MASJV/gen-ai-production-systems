#Base Image
FROM python:3.11-slim

#Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

#Print logs immediately
ENV PYTHONUNBUFFERED=1

#Working directory
WORKDIR /app

#Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

#Copy requirements first
COPY requirements.txt .

#Upgrade pip
RUN pip install --upgrade pip

#Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

#Copy project files
COPY . .

#Streamlit Port
EXPOSE 8501

#Streamlit Configuration
RUN mkdir -p /root/.streamlit

RUN echo "\
[server]\n\
headless = true\n\
port = 8501\n\
address = '0.0.0.0'\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml

#Start ENV
CMD ["streamlit", "run", "app.py"]