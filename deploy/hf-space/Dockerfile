# Lunar plate solver API for a Hugging Face Docker Space (CPU).
# Built by scripts/deploy_hf.py, which copies lunarsolve/, server/, weights/ and this folder
# into one directory and uploads it to the Space.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    LUNARSOLVE_DATA=/home/user/data \
    OMP_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
# Bake the data (about 1 GB) into the image: faster cold starts than a download at every start.
RUN python server/fetch_data.py --repo "$(cat data_repo.txt)"

EXPOSE 7860
CMD ["python", "server/api.py", "--host", "0.0.0.0", "--port", "7860"]
