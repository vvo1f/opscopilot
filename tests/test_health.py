from fastapi.testclient import TestClient

from opscopilot.main import app


def test_health_has_both_db_flags():
    # no database runs in the unit-test environment, so values are False;
    # we only check that the endpoint answers and has the right shape
    body = TestClient(app).get("/health").json()
    assert set(body) == {"app_db", "geo_db"}
