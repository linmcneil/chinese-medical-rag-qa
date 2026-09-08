# 本地 CPU 演示版（无需 GPU）。
# 构建：docker compose up --build   或   docker build -t ragqa-cpu .
# 首次启动会用内置样例自动建索引；问答可选用 CPU 小模型（默认 Qwen2.5-0.5B，约 1GB）。
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 先装依赖（利用 Docker 层缓存）
COPY requirements-cpu.txt ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \\
    && pip install -r requirements-cpu.txt

COPY . .

# 纯逻辑静态检查：不下载模型、不引入重型依赖
RUN python -m compileall -q ragqa scripts qa_system.py app_demo.py app_local.py document_processer.py

EXPOSE 7861

# 首次启动自动建索引，然后打开本地 CPU 版网页（0.0.0.0 供容器端口映射）
ENTRYPOINT ["bash", "docker_entrypoint.sh"]