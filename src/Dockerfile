ARG SKIP_FRONTEND=false
ARG COMMON_IMAGE_TAG=tero-common

FROM ${COMMON_IMAGE_TAG} AS frontend-builder
ARG SKIP_FRONTEND
ARG VITE_APP_INSIGHTS_CONNECTION

WORKDIR /app/frontend
COPY frontend/package.json ./

RUN if [ "$SKIP_FRONTEND" = "false" ]; then pnpm install --frozen-lockfile; fi

COPY frontend/ ./

ENV VITE_APP_INSIGHTS_CONNECTION=$VITE_APP_INSIGHTS_CONNECTION

# Run type-check and vite build sequentially to avoid OOM in constrained Docker environments.
# NODE_OPTIONS gives Node.js a 4 GB heap ceiling (default is ~1.7 GB).
RUN if [ "$SKIP_FRONTEND" = "false" ]; then \
      NODE_OPTIONS="--max-old-space-size=4096" pnpm run type-check && \
      NODE_OPTIONS="--max-old-space-size=4096" pnpm run build-only; \
    else mkdir dist; fi

FROM python:3.13

RUN apt-get update && apt-get install -y netcat-traditional

# This avoids unnecessary warning from transformers library due to missing pytorch and other libraries that are not really necessary.
ENV TRANSFORMERS_VERBOSITY=error

RUN pip install poetry

RUN poetry config installer.max-workers 10

WORKDIR /usr/src/app
ENV PYTHONPATH=/usr/src/app

COPY docker/wait-for-it.sh wait-for-it.sh
COPY backend/pyproject.toml backend/poetry.lock ./

RUN poetry install --no-root

COPY docker/entrypoint.sh entrypoint.sh
COPY sample.env .env
COPY backend/alembic.ini alembic.ini
COPY backend/alembic ./alembic
COPY backend/tero ./tero
COPY --from=frontend-builder /app/frontend/dist ./frontend

RUN chmod +x wait-for-it.sh
ENV FRONTEND_PATH=frontend

ENTRYPOINT [ "./entrypoint.sh" ]

EXPOSE 8000

CMD ["poetry", "run", "python", "-m", "tero"]
