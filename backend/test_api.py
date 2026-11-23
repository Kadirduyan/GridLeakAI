"""Gerçek proje veritabanına dokunmadan API davranışlarını doğrular."""
import pytest
from fastapi.testclient import TestClient
from .main import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "test.sqlite3")) as api:
        yield api


def test_demo_data_and_missing_model(client):
    assert client.get('/api/health').json()['database'] == 'ok'
    response = client.get('/api/subscribers').json()
    assert response['total'] == 3
    assert all(s['model_score'] is None for s in response['items'])
    assert client.get('/api/subscribers?q=104829').json()['total'] == 1
    assert client.get('/api/subscribers?offset=3').json()['items'] == []
    assert client.get('/api/subscribers/unknown').status_code == 404
    assert client.get('/api/model/status').json()['status'] == 'not_connected'


def test_consumption_preserves_missing_and_zero(client):
    data = client.get('/api/subscribers/AB-104829/consumption?days=365').json()
    points = data['points']
    assert len(points) == 365
    assert sum(p['consumption_kwh'] is None for p in points) == 7
    assert sum(p['consumption_kwh'] == 0 for p in points) == 11
    assert client.get('/api/subscribers/AB-104829/consumption?days=0').status_code == 422


def test_plan_capacity_quality_and_idempotency(client):
    assert client.put('/api/plan/capacity',json={'capacity':1}).status_code == 200
    assert client.put('/api/plan/AB-104829').status_code == 200
    assert client.put('/api/plan/AB-104829').json()['status'] == 'already_planned'
    assert client.put('/api/plan/AB-108173').status_code == 409
    assert client.put('/api/plan/AB-112406').status_code == 409
    assert client.put('/api/plan/capacity',json={'capacity':2}).status_code == 200
    assert client.put('/api/plan/AB-108173').status_code == 200
    assert client.put('/api/plan/capacity',json={'capacity':1}).status_code == 409
    assert len(client.get('/api/plan').json()['items']) == 2


def test_validation_and_result_rules(client):
    assert client.put('/api/plan/capacity',json={'capacity':True}).status_code == 422
    assert client.put('/api/plan/capacity',json={'capacity':0}).status_code == 422
    assert client.put('/api/subscribers/AB-104829/note',json={'note':'x'*5001}).status_code == 422
    assert client.put('/api/results/AB-104829',json={'status':'Geçersiz','note':''}).status_code == 422
    assert client.put('/api/results/AB-104829',json={'status':'Planlandı','note':''}).status_code == 409
    assert client.put('/api/subscribers/unknown/note',json={'note':'test'}).status_code == 404
    assert client.put('/api/subscribers/AB-104829/note',json={'note':'test','FLAG':1}).status_code == 422


def test_persistence_after_restart_and_removal(tmp_path):
    path = tmp_path / 'persistent.sqlite3'
    with TestClient(create_app(path)) as api:
        api.put('/api/plan/AB-104829')
        api.put('/api/subscribers/AB-104829/note',json={'note':"Sayaç kontrolü; ' özel karakter"})
        assert api.put('/api/results/AB-104829',json={'status':'Kaçak bulunmadı','note':'Örnek sonuç'}).status_code == 200
        api.delete('/api/plan/AB-104829')
    with TestClient(create_app(path)) as api:
        state = api.get('/api/state').json()
        assert state['plan'] == []
        assert state['notes'][0]['note'] == "Sayaç kontrolü; ' özel karakter"
        assert state['results'][0]['status'] == 'Kaçak bulunmadı'
        assert api.put('/api/results/AB-104829',json={'status':'Ek inceleme gerekli','note':'Takip'}).status_code == 200


def test_cors(client):
    headers = {'Origin':'http://localhost:5173','Access-Control-Request-Method':'PUT'}
    response = client.options('/api/plan/AB-104829',headers=headers)
    assert response.headers['access-control-allow-origin'] == 'http://localhost:5173'
    headers['Origin'] = 'https://unrelated.example'
    assert 'access-control-allow-origin' not in client.options('/api/plan/AB-104829',headers=headers).headers


def test_built_frontend_and_api_share_server(client):
    import re
    page = client.get('/')
    assert page.status_code == 200
    assets = re.findall(r'(?:src|href)="(\.?/assets/[^"]+)"', page.text)
    assert assets
    for asset in assets:
        assert client.get(asset.removeprefix('.')).status_code == 200
    assert client.get('/backend/database.py').status_code == 404
    assert client.get('/api/health').json()['database'] == 'ok'
