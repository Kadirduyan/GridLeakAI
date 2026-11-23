"""GridLeak AI — Salt-Okunur AI Katalog Okuyucu (`catalog.sqlite3`).

AI dışa aktarımının ürettiği SQLite kataloğunu salt-okunur (`mode=ro`) açar; sentetik veri üretmez.
- Abone özetlerini, kalibre risk skorlarını, sıralamayı ve veri doluluk oranını döner.
- Seçilen abonenin günlük tüketim serisini ve 7 günlük hareketli ortalamasını hesaplar.
- İlk 2.000 yüksek riskli abone için kaydedilmiş TreeSHAP özellik katkılarını sunar.
"""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from fastapi import HTTPException

SOURCE='sgcc'
class Catalog:
    def __init__(self,path): self.path=Path(path).resolve()
    @contextmanager
    def connection(self):
        if not self.path.is_file():
            raise HTTPException(503,'AI veri kataloğu hazır değil. Eğitim, değerlendirme ve dışa aktarımı tamamlayın.')
        db=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
        try: yield db
        finally: db.close()
    def status(self):
        if not self.path.is_file():return {'status':'not_connected','source':SOURCE,'message':'AI kataloğu henüz dışa aktarılmadı.','metrics':[],'cv':[]}
        with self.connection() as db:
            data=json.loads(db.execute("SELECT value FROM metadata WHERE key='catalog'").fetchone()[0])
        data.pop('dates',None)
        return data
    def require_subscriber(self,subscriber_id):
        with self.connection() as db:
            row=db.execute('SELECT * FROM subscribers WHERE id=?',(subscriber_id,)).fetchone()
        if row is None:raise HTTPException(404,'Abone bulunamadı.')
        return row
    @staticmethod
    def serialize(row):
        return {'id':row['id'],'source':SOURCE,'total_days':row['total'],'recorded_days':row['recorded'],
            'missing_days':row['total']-row['recorded'],'data_quality_percent':row['quality'],
            'model_score':row['score'],'inspection_rank':row['rank'],'partition':row['partition'],
            'prediction':row['prediction'],'model_status':'connected' if row['score'] is not None else 'insufficient_data',
            'review_action':'data_check' if row['quality']<90 else 'analyst_review'}
    def summary(self,subscriber_id):return self.serialize(self.require_subscriber(subscriber_id))
    def list(self,q='',offset=0,limit=100):
        term='%'+q.strip().replace('\\','\\\\').replace('%','\\%').replace('_','\\_')+'%'
        with self.connection() as db:
            total=db.execute("SELECT COUNT(*) FROM subscribers WHERE id LIKE ? ESCAPE '\\'",(term,)).fetchone()[0]
            rows=db.execute("SELECT id,score,rank,quality,recorded,total,partition,prediction FROM subscribers WHERE id LIKE ? ESCAPE '\\' ORDER BY rank IS NULL, rank, id LIMIT ? OFFSET ?",(term,limit,offset)).fetchall()
        return {'items':[self.serialize(r) for r in rows],'total':total,'offset':offset,'limit':limit,'source':SOURCE}
    def consumption(self,subscriber_id,days):
        row=self.require_subscriber(subscriber_id);values=json.loads(row['points'])
        with self.connection() as db:dates=json.loads(db.execute("SELECT value FROM metadata WHERE key='catalog'").fetchone()[0])['dates']
        points=[]
        for i in range(max(0,len(values)-days),len(values)):
            window=[v for v in values[max(0,i-6):i+1] if v is not None]
            points.append({'date':dates[i],'consumption_kwh':values[i],
                'moving_average_7d':sum(window)/len(window) if window else None})
        return {'subscriber_id':subscriber_id,'source':SOURCE,'unit':'kWh/day','points':points}
    def explanation(self,subscriber_id):
        row=self.require_subscriber(subscriber_id);data=json.loads(row['explanation'])
        return {'subscriber_id':subscriber_id,'status':'ready' if data else 'not_computed',
            'explanation':data,'feature_values':json.loads(row['features']),
            'message':'Model katkıları kaçak kanıtı değildir.' if data else 'Bu abone için SHAP henüz hesaplanmadı; etki puanı üretilmez.'}
