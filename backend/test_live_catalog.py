"""GridLeak AI — Canlı AI Kataloğu (`catalog.sqlite3`) Entegrasyon Testi (7. Test).

Aktif v3 dışa aktarımının ürettiği gerçek `catalog.sqlite3` üzerinde salt-okunur olarak:
model bağlantı durumu, metrik varlığı, skor sıralamasının monotonluğu, `FLAG` gizliliği
ve 2016 günlük tüketim serisi bütünlüğünü doğrular.
"""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from .main import create_app
CATALOG=Path(__file__).resolve().parents[1]/'data/application/catalog.sqlite3'
@pytest.mark.skipif(not CATALOG.is_file(),reason='Gerçek AI dışa aktarımı henüz tamamlanmadı')
def test_real_catalog(tmp_path):
    with TestClient(create_app(tmp_path/'state.sqlite3',CATALOG)) as c:
        status=c.get('/api/model/status').json()
        assert status['status']=='connected' and status['source']=='sgcc'
        assert status['metrics'] and status['scored']<=status['total']
        page=c.get('/api/subscribers?limit=10').json()
        assert page['total']==status['total']
        scores=[x['model_score'] for x in page['items']]
        assert scores==sorted(scores,reverse=True)
        for row in page['items']:
            assert 'FLAG' not in row
            data=c.get(f"/api/subscribers/{row['id']}/consumption?days=365").json()['points']
            assert len(data)==row['total_days']
            assert sum(x['consumption_kwh'] is not None for x in data)==row['recorded_days']
            assert any(x['date']=='2016-09-18' and x['consumption_kwh'] is None for x in data)
