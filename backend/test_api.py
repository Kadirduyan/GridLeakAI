"""GridLeak AI — Backend API Sözleşme ve Güvenlik Testleri (6 Test).

Üretim veritabanlarına dokunmadan geçici SQLite katalogları üzerinde:
abone arama/gizlilik (`FLAG` yok), eksik/sıfır tüketim, denetim planı kalıcılığı,
Pydantic 422 doğrulaması, eksik katalog 503 davranışı ve statik dosya sunumunu sınar.
"""
import json
import sqlite3
import pytest
from fastapi.testclient import TestClient
from .main import create_app

@pytest.fixture
def paths(tmp_path):
    catalog=tmp_path/'catalog.sqlite3'
    with sqlite3.connect(catalog) as db:
        db.execute('CREATE TABLE metadata(key TEXT,value TEXT)')
        db.execute('INSERT INTO metadata VALUES (?,?)',('catalog',json.dumps({'status':'connected','source':'sgcc','dates':['2016-01-01','2016-01-02','2016-01-03']})))
        db.execute('CREATE TABLE subscribers(id TEXT PRIMARY KEY,score REAL,rank INTEGER,quality REAL,recorded INTEGER,total INTEGER,partition TEXT,prediction INTEGER,points TEXT,features TEXT,explanation TEXT)')
        db.executemany('INSERT INTO subscribers VALUES (?,?,?,?,?,?,?,?,?,?,?)',[
            ('A',.8,1,100,3,3,'test',1,'[0,2,3]','{}','null'),
            ('B',.2,2,66.67,2,3,'training',0,'[0,null,3]','{}','null')])
    return tmp_path/'state.sqlite3',catalog

@pytest.fixture
def client(paths):
    with TestClient(create_app(*paths)) as c:yield c

def test_catalog_search_and_privacy(client):
    assert client.get('/api/health').json()['model_status']=='connected'
    data=client.get('/api/subscribers?limit=1').json()
    assert data['total']==2 and data['items'][0]['id']=='A'
    assert 'FLAG' not in data['items'][0]
    assert client.get('/api/subscribers?q=%').json()['total']==0
    assert client.get('/api/subscribers?q=B').json()['items'][0]['partition']=='training'
    assert client.get('/api/subscribers/unknown').status_code==404

def test_missing_zero_and_explanation(client):
    p=client.get('/api/subscribers/B/consumption').json()['points']
    assert p[0]['consumption_kwh']==0 and p[1]['consumption_kwh'] is None
    assert client.get('/api/subscribers/A/explanation').json()['status']=='not_computed'

def test_plan_and_persistence(paths):
    with TestClient(create_app(*paths)) as c:
        assert c.put('/api/plan/A').status_code==200
        assert c.put('/api/plan/B').status_code==409
        assert c.put('/api/subscribers/A/note',json={'note':'Kontrol'}).status_code==200
        assert c.put('/api/results/A',json={'status':'Kaçak bulunmadı','note':'Saha'}).status_code==200
    with TestClient(create_app(*paths)) as c:
        assert c.get('/api/state').json()['notes'][0]['note']=='Kontrol'

def test_validation(client):
    assert client.put('/api/plan/capacity',json={'capacity':True}).status_code==422
    assert client.get('/api/subscribers/A/consumption?days=0').status_code==422
    assert client.put('/api/subscribers/A/note',json={'note':'x','FLAG':1}).status_code==422

def test_missing_catalog(tmp_path):
    with TestClient(create_app(tmp_path/'state.sqlite3',tmp_path/'absent.sqlite3')) as c:
        assert c.get('/api/model/status').json()['status']=='not_connected'
        assert c.get('/api/subscribers').status_code==503

def test_static_serving(client):
    assert client.get('/').status_code==200
    assert client.get('/backend/main.py').status_code==404
