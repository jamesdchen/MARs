# Polymarket order book recorder.
#   docker build -t polyrec .
#   docker run -d --name polyrec --restart unless-stopped --stop-timeout 90 \
#       -v /var/lib/polyrec:/data polyrec
# (or deploy/docker-compose.yml). Output, status.json and logs land in the /data volume.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN useradd --system --uid 10001 --home /app polyrec \
    && mkdir -p /data && chown polyrec /data

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY polyrec ./polyrec
COPY backfill_winners.py verify_output.py config.example.toml ./

USER polyrec
VOLUME ["/data"]
STOPSIGNAL SIGTERM

# healthy = status.json refreshed within 60 s and market data seen within 5 min
HEALTHCHECK --interval=60s --timeout=10s --start-period=180s --retries=3 \
  CMD python -c "import json,time,sys; s=json.load(open('/data/status.json')); sys.exit(0 if time.time()-s['time']<60 and s['ws']['last_frame_age_s']<300 else 1)"

ENTRYPOINT ["python", "-m", "polyrec"]
CMD ["record", "--config", "/app/config.example.toml", "--out", "/data"]
