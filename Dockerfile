FROM pytorch-notebook-cpu-stata:latest
WORKDIR /app
COPY . .
RUN pip install -e .