"""OpenAQ request shape.

The failure mode these pin is silent: a mis-encoded parameter returns HTTP 200
with a full page of locations that then get filtered to nothing, so the pipeline
reports success with zero rows.
"""

from __future__ import annotations

from datetime import UTC

import httpx
import pytest

from aqetl import openaq
from aqetl.config import CITIES, PARAMETERS, SEARCH_RADIUS_M


def _capture(monkeypatch) -> list[httpx.URL]:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json={"meta": {"found": 0}, "results": []})

    monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
    real = httpx.Client

    def patched(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(openaq.httpx, "Client", patched)
    return seen


def test_parameters_id_is_repeated_not_comma_joined(monkeypatch):
    """OpenAQ types parameters_id as array<int>; a joined string is ignored."""
    seen = _capture(monkeypatch)
    with httpx.Client() as client:
        openaq.fetch_locations(client, CITIES[0])

    assert len(seen) == 1
    repeated = seen[0].params.get_list("parameters_id")
    assert sorted(int(v) for v in repeated) == sorted(PARAMETERS)
    assert not any("," in v for v in repeated)


def test_coordinates_and_radius_are_sent(monkeypatch):
    seen = _capture(monkeypatch)
    city = CITIES[0]
    with httpx.Client() as client:
        openaq.fetch_locations(client, city)

    params = seen[0].params
    assert params["coordinates"] == f"{city.latitude},{city.longitude}"
    assert int(params["radius"]) == SEARCH_RADIUS_M


def test_missing_key_raises_before_any_request(monkeypatch):
    calls: list[httpx.Request] = []
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    real = httpx.Client

    def patched(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(
            lambda r: calls.append(r) or httpx.Response(200, json={"results": []})
        )
        return real(*args, **kwargs)

    monkeypatch.setattr(openaq.httpx, "Client", patched)
    with httpx.Client() as client, pytest.raises(RuntimeError, match="OPENAQ_API_KEY"):
        openaq.fetch_locations(client, CITIES[0])
    assert calls == []


def test_mobile_locations_are_skipped_and_rows_are_typed(monkeypatch):
    """A city with only mobile units must yield an empty but correctly typed frame."""
    monkeypatch.setenv("OPENAQ_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/locations"):
            return httpx.Response(
                200,
                json={
                    "meta": {"found": 1},
                    "results": [
                        {
                            "id": 1,
                            "name": "Bus",
                            "isMobile": True,
                            "sensors": [{"id": 5, "parameter": {"id": 2}}],
                        }
                    ],
                },
            )
        return httpx.Response(200, json={"results": []})

    real = httpx.Client

    def patched(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(openaq.httpx, "Client", patched)
    from datetime import datetime

    start = datetime(2026, 11, 1, tzinfo=UTC)
    frame = openaq.fetch_all(start, start, cities=(CITIES[0],))
    assert frame.is_empty()
    assert dict(frame.schema) == openaq.EXTRACT_SCHEMA


def test_retired_stations_are_not_selected_over_live_ones(monkeypatch):
    """OpenAQ keeps retired stations; picking by id used to select decade-dead ones.

    This is the regression behind a run that reported 'extract 0 rows' while a
    station in the same city was reporting that hour.
    """
    from datetime import datetime

    monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
    dead = {
        "id": 18,
        "name": "Dead",
        "isMobile": False,
        "datetimeLast": {"utc": "2016-11-09T16:00:00Z"},
        "sensors": [{"id": 37, "parameter": {"id": 2, "units": "µg/m³"}}],
    }
    live = {
        "id": 4946812,
        "name": "Live",
        "isMobile": False,
        "datetimeLast": {"utc": "2026-11-18T07:05:00Z"},
        "sensors": [{"id": 900, "parameter": {"id": 2, "units": "µg/m³"}}],
    }
    queried: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/locations"):
            return httpx.Response(200, json={"meta": {"found": 2}, "results": [dead, live]})
        sensor = int(path.split("/sensors/")[1].split("/")[0])
        queried.append(sensor)
        end = "2026-11-18T08:00:00Z"
        return httpx.Response(
            200,
            json={
                "meta": {"limit": 1000},
                "results": [{
                    "value": 42.0,
                    "parameter": {"units": "µg/m³"},
                    "period": {"datetimeFrom": {"utc": "2026-11-18T07:00:00Z"},
                               "datetimeTo": {"utc": end}},
                    "coverage": {"percentComplete": 100.0},
                }],
            },
        )

    real = httpx.Client

    def patched(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(openaq.httpx, "Client", patched)

    start = datetime(2026, 11, 18, 0, tzinfo=UTC)
    frame = openaq.fetch_all(start, datetime(2026, 11, 18, 9, tzinfo=UTC),
                             cities=(CITIES[0],))
    assert queried == [900], f"queried retired sensors: {queried}"
    assert len(frame) == 1
    assert frame["station_name"].to_list() == ["Live"]


def test_transient_500_is_retried(monkeypatch):
    """One 500 killed a whole scheduled run; 5xx is transient and must retry."""
    monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
    monkeypatch.setattr(openaq, "_BACKOFF_S", 0.0)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(500, json={"message": "boom"})
        return httpx.Response(200, json={"meta": {"found": 0}, "results": []})

    real = httpx.Client

    def patched(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(openaq.httpx, "Client", patched)
    with httpx.Client() as client:
        payload = openaq._request(client, f"{openaq.OPENAQ_BASE}/locations", {})
    assert payload["results"] == []
    assert len(calls) == 2


def test_401_is_not_retried(monkeypatch):
    """A bad key must fail immediately, not burn four requests."""
    monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"message": "unauthorized"})

    real = httpx.Client

    def patched(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(openaq.httpx, "Client", patched)
    with httpx.Client() as client, pytest.raises(RuntimeError, match="OPENAQ_API_KEY"):
        openaq._request(client, f"{openaq.OPENAQ_BASE}/locations", {})
    assert len(calls) == 1


def test_transport_error_is_retried(monkeypatch):
    """REGRESSION: only HTTP statuses retried, so one flaky handshake aborted."""
    monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
    monkeypatch.setattr(openaq, "_BACKOFF_S", 0.0)
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectTimeout("handshake timed out")
        return httpx.Response(200, json={"results": []})

    real = httpx.Client

    def patched(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(openaq.httpx, "Client", patched)
    with httpx.Client() as client:
        payload = openaq._request(client, f"{openaq.OPENAQ_BASE}/locations", {})
    assert payload["results"] == []
    assert len(attempts) == 2


def test_429_sleep_is_capped(monkeypatch):
    """REGRESSION: an epoch-valued reset header would sleep for 20,000 days."""
    epoch = httpx.Response(429, headers={"x-ratelimit-reset": "1763900000"})
    assert openaq._reset_seconds(epoch) == openaq._MAX_SLEEP_S
    fine = httpx.Response(429, headers={"x-ratelimit-reset": "30"})
    assert openaq._reset_seconds(fine) == 30
