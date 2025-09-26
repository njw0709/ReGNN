FROM pytorch-notebook-gpu-stata:latest
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
# RUN pip install -e