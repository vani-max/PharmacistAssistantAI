# Stage 1: Build the React frontend
FROM node:20 AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Serve with Python backend
FROM python:3.10-slim

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment and grading packages
COPY env/ env/
COPY graders/ graders/
COPY server/ server/
COPY agent/ agent/
COPY rl_weights/ rl_weights/

# Copy root-level files required by OpenEnv
COPY inference.py .
COPY openenv.yaml .
COPY pyproject.toml .

# Copy the built frontend from Stage 1
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist

# Expose the FastAPI server port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]