# Wallex trading bot for Liara (DEPLOY.md). Secrets come from Liara environment variables, never from the image.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8080 BOT_PERSIST=/app/persist TZ=UTC
# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# wireproxy: userspace WireGuard client exposing a local HTTP proxy (no root / NET_ADMIN / TUN needed), used to
# reach Binance from Iran (deploy/tunnel.py). Pinned release, verified against its published SHA-256.
ARG WIREPROXY_URL=https://github.com/windtf/wireproxy/releases/download/v1.1.3/wireproxy_linux_amd64.tar.gz
ARG WIREPROXY_SHA256=e88c1d090740373fc606c1bafd81d9a5eadc642cce5667616e20e9d7a444f51c
RUN python -c "import urllib.request,hashlib,tarfile,io,os; b=urllib.request.urlopen('${WIREPROXY_URL}',timeout=180).read(); h=hashlib.sha256(b).hexdigest(); assert h=='${WIREPROXY_SHA256}', 'wireproxy checksum mismatch: '+h; t=tarfile.open(fileobj=io.BytesIO(b)); m=[x for x in t.getmembers() if os.path.basename(x.name)=='wireproxy'][0]; open('/usr/local/bin/wireproxy','wb').write(t.extractfile(m).read()); os.chmod('/usr/local/bin/wireproxy',0o755)"

COPY cli.py config.yaml blocked_markets.txt ./
COPY engine/ engine/
COPY deploy/ deploy/
COPY models/latest.joblib models/latest.joblib
COPY agent_cache/ agent_cache/
COPY data/fed_calendar.yaml data/fed_calendar.yaml

EXPOSE 8080
CMD ["python", "deploy/start.py"]
