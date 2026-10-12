# Lunar plate solver API, CPU image for any Linux server (x86-64 or ARM64).
# Build context: the repository root (see docker-compose.yml).
FROM python:3.12-slim

RUN useradd -m -u 1000 app
USER app
ENV HOME=/home/app \
    PATH=/home/app/.local/bin:$PATH \
    LUNARSOLVE_DATA=/home/app/data \
    PYTHONUNBUFFERED=1
WORKDIR /home/app/srv

COPY --chown=app deploy/server/requirements.txt requirements.txt
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

# Data (about 1 GB) from the Hugging Face dataset, in its own layer: code changes do not
# download it again.
ARG DATA_REPO=turtlely/lunarsolve-data
COPY --chown=app server/fetch_data.py server/fetch_data.py
RUN python server/fetch_data.py --repo "$DATA_REPO"

COPY --chown=app lunarsolve lunarsolve
COPY --chown=app server server
COPY --chown=app weights/craternet.pt weights/craternet.pt

EXPOSE 8000
CMD ["python", "server/api.py", "--host", "0.0.0.0", "--port", "8000"]
