"""Gerçek SGCC dosyasından bağımsız, açıkça etiketlenmiş örnek veri sağlayıcısı."""
import math
from datetime import date, timedelta
from fastapi import HTTPException

SOURCE = "synthetic_demo"
SUBSCRIBERS = {
    "AB-104829": {"missing": [39,118,205,244,302,320,348], "drop":64, "zero":11},
    "AB-108173": {"missing": [19,55,118,180,220,260,301,340,350,351,352,353], "drop":38, "zero":4},
    "AB-112406": {"missing": list(range(0,292,2)), "drop":51, "zero":7},
}


def require_subscriber(subscriber_id):
    if subscriber_id not in SUBSCRIBERS:
        raise HTTPException(404, "Abone bulunamadı.")
    return SUBSCRIBERS[subscriber_id]


def summary(subscriber_id):
    record = require_subscriber(subscriber_id)
    valid = 365 - len(record["missing"])
    return {
        "id": subscriber_id, "source": SOURCE,
        "total_days":365, "recorded_days":valid,
        "missing_days":len(record["missing"]),
        "data_quality_percent": round(valid / 365 * 100, 2),
        "model_score":None, "inspection_rank":None,
        "model_status":"not_connected",
        "review_action":"manual_review" if valid / 365 >= .9 else "data_check",
    }


def consumption(subscriber_id, days):
    record = require_subscriber(subscriber_id)
    values = []
    for i in range(365):
        if i in record["missing"]:
            value = None
        elif 330 <= i < 330 + record["zero"]:
            value = 0.0
        else:
            base = 16 * (1 - record["drop"] / 100) if i > 300 else 16
            value = max(0, base + math.sin(i*.19)*2 + math.cos(i*1.31)*1.5 + (3 if i%7==0 else 0))
        values.append(value)
    points = []
    for i in range(365-days,365):
        window = [v for v in values[max(0,i-6):i+1] if v is not None]
        points.append({
            "date": (date(2016,1,1)+timedelta(days=i)).isoformat(),
            "consumption_kwh": round(values[i],3) if values[i] is not None else None,
            "moving_average_7d": round(sum(window)/len(window),3) if window else None,
        })
    return {"subscriber_id":subscriber_id,"source":SOURCE,"unit":"kWh/day","points":points}
