FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY ragqa ./ragqa
COPY examples ./examples
RUN pip install --no-cache-dir ".[api]" && useradd --create-home appuser \
    && mkdir /app/artifacts && chown appuser:appuser /app/artifacts
USER appuser
RUN python -m ragqa build --data examples/knowledge.json --index artifacts/demo
EXPOSE 8000
CMD ["python", "-m", "ragqa", "serve", "--index", "artifacts/demo", "--host", "0.0.0.0"]
