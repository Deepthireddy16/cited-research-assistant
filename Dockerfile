# python:3.11-slim: a stripped-down base image — smaller and faster to
# build/pull than the full python:3.11 image, since it omits docs, extra
# compilers, and GUI libraries we don't need for this project.
FROM python:3.11-slim

# Set the working directory inside the container — all subsequent
# commands (COPY, RUN, CMD) run relative to this path.
WORKDIR /app

# Copy just requirements.txt first, before the rest of the code.
# Why: Docker caches each layer. If only your source code changes (not
# your dependencies), Docker reuses the cached "pip install" layer
# instead of re-running it — much faster rebuilds during development.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project code.
COPY src/ ./src/

# Streamlit's default port — exposing it documents which port the
# container listens on (doesn't actually publish it; that happens in
# docker-compose.yml).
EXPOSE 8501

# Run the Streamlit app when the container starts.
# --server.address=0.0.0.0 is required: Streamlit defaults to
# localhost, which would only be reachable from INSIDE the container.
# 0.0.0.0 makes it listen on all network interfaces so Docker can
# forward traffic to it from outside.
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
