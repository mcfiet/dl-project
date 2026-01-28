# syntax=docker/dockerfile:1

FROM node:20-alpine AS ui-builder
WORKDIR /app/ui
COPY ui/ ./
RUN if [ -f package-lock.json ]; then npm ci; \
  elif [ -f pnpm-lock.yaml ]; then corepack enable && pnpm install; \
  elif [ -f yarn.lock ]; then corepack enable && yarn install --frozen-lockfile; \
  else npm install; fi
RUN npm run build

FROM nginx:1.25-alpine AS ui
COPY ui/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=ui-builder /app/ui/dist /usr/share/nginx/html

FROM python:3.11-slim AS api
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY scripts ./scripts
COPY models ./models
EXPOSE 8000
CMD ["python","-m","uvicorn","scripts.serve_points_scored:app","--host","0.0.0.0","--port","8000"]
