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
