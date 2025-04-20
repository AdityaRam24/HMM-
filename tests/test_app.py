import pytest
from app import app

@pytest.fixture
def client():
    # Configure Flask for testing
    app.config['TESTING'] = True
    return app.test_client()

def test_index_route(client):
    """GET / should return 200 and contain the form"""
    resp = client.get('/')
    assert resp.status_code == 200
    # adjust this assertion to match something in your index.html
    assert b'<form' in resp.data  

def test_api_tickers_no_query(client):
    """GET /api/tickers with no q should return full list"""
    resp = client.get('/api/tickers')
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    # SPY is in your common_tickers dict
    assert any(item['symbol'] == 'SPY' for item in data)

def test_api_tickers_with_query(client):
    """GET /api/tickers?q=A returns only symbols starting with 'A'"""
    resp = client.get('/api/tickers?q=A')
    assert resp.status_code == 200
    data = resp.get_json()
    assert all(item['symbol'].startswith('A') for item in data)
