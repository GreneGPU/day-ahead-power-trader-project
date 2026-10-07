from fastapi.testclient import TestClient
import pytest

from api.index import app


client = TestClient(app)


@pytest.mark.parametrize("host", ["https://eu.i.posthog.com", "https://us.i.posthog.com"])
def test_production_analytics_exposes_only_public_config(monkeypatch, host):
    monkeypatch.setenv("VERCEL_ENV", "production")
    monkeypatch.setenv("POSTHOG_PROJECT_TOKEN", "phc_test_project")
    monkeypatch.setenv("POSTHOG_HOST", host + "/")
    monkeypatch.setenv("POSTHOG_PERSONAL_API_KEY", "private-must-not-be-exposed")
    response = client.get("/api/analytics-config")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {"enabled": True, "token": "phc_test_project", "host": host}


@pytest.mark.parametrize("environment", ["preview", "development", ""])
def test_nonproduction_never_enables_analytics(monkeypatch, environment):
    monkeypatch.setenv("VERCEL_ENV", environment)
    monkeypatch.setenv("POSTHOG_PROJECT_TOKEN", "phc_test_project")
    monkeypatch.setenv("POSTHOG_HOST", "https://eu.i.posthog.com")
    assert client.get("/api/analytics-config").json() == {"enabled": False}


@pytest.mark.parametrize("token,host", [
    ("", "https://eu.i.posthog.com"),
    ("phx_personal_key", "https://eu.i.posthog.com"),
    ("phc_test_project", ""),
    ("phc_test_project", "https://untrusted.example"),
])
def test_missing_or_invalid_configuration_disables_analytics(monkeypatch, token, host):
    monkeypatch.setenv("VERCEL_ENV", "production")
    monkeypatch.setenv("POSTHOG_PROJECT_TOKEN", token)
    monkeypatch.setenv("POSTHOG_HOST", host)
    assert client.get("/api/analytics-config").json() == {"enabled": False}
