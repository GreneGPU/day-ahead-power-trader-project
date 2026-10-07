"""Read-only, cached DK1 published day-ahead prices."""
from datetime import datetime, timedelta, timezone
import json
import math
from threading import Lock
import time
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

_cache = None
_expires = 0
_lock = Lock()


def latest_prices():
    global _cache, _expires
    with _lock:
        if _cache is not None and time.monotonic() < _expires:
            return _cache
        now = datetime.now(timezone.utc)
        today = now.astimezone(ZoneInfo("Europe/Copenhagen")).date()
        url = "https://api.energidataservice.dk/dataset/DayAheadPrices?" + urlencode({
            "start": today.isoformat(), "end": (today + timedelta(days=2)).isoformat(),
            "filter": json.dumps({"PriceArea": ["DK1"]}), "sort": "TimeUTC asc", "limit": 300,
        })
        with urlopen(url, timeout=15) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("Unexpectedly large price response.")
        data = json.loads(raw)
        rows = []
        for row in data["records"]:
            price = float(row["DayAheadPriceDKK"])
            if row["PriceArea"] != "DK1" or not math.isfinite(price):
                raise ValueError("Invalid price response.")
            stamp = datetime.fromisoformat(row["TimeUTC"]).replace(tzinfo=timezone.utc)
            rows.append({"time_utc": stamp.isoformat(), "price_dkk_mwh": price})
        _cache = {"fetched_at": now.isoformat(), "area": "DK1", "rows": sorted(rows, key=lambda row: row["time_utc"])}
        _expires = time.monotonic() + 600
        return _cache
