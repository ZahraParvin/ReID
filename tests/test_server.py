"""Real PostgreSQL API tests. DATABASE_URL must point to a disposable *_test database."""

import importlib
import os
from urllib.parse import urlparse

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.database


@pytest.fixture(scope="module")
def server_module():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        pytest.skip("Set DATABASE_URL to a dedicated *_test PostgreSQL database")
    if not urlparse(dsn).path.removeprefix("/").endswith("_test"):
        pytest.fail("Refusing to reset a database whose name does not end in _test")
    module = importlib.import_module("server")
    module.initialize_db()
    yield module
    module.db_pool.closeall()


@pytest.fixture
def client(server_module):
    with TestClient(server_module.app) as client:
        client.post("/gallery/reset").raise_for_status()
        server_module.THRESHOLDS.update(person=0.6, vehicle=0.5)
        yield client
        client.post("/gallery/reset").raise_for_status()


def track(client, features, camera="one", category=2):
    response = client.post(
        "/register_track",
        json={"camera_id": camera, "yolo_class": category, "features": features},
    )
    response.raise_for_status()
    return response.json()


def test_matching_and_distinct_identity(client):
    first = track(client, [1, 0, 0])
    same = track(client, [1, 0, 0], camera="two")
    different = track(client, [0, 1, 0])
    assert first["is_new"] and not same["is_new"] and different["is_new"]
    assert first["global_id"] == same["global_id"] != different["global_id"]
    sightings = client.get(f"/identities/{first['global_id']}/sightings").json()[
        "sightings"
    ]
    assert {s["camera_id"] for s in sightings} == {"one", "two"}


def test_gallery_reload_preserves_features_and_matching(client, server_module):
    first = track(client, [1, 0, 0])
    track(client, [0.9, 0.1, 0])
    before = server_module.gallery_db["vehicle"][0]["features"].copy()
    for category in server_module.gallery_db:
        server_module.gallery_db[category].clear()
        server_module.id_counters[category] = 0
    server_module.load_gallery_from_db()
    np.testing.assert_array_equal(
        server_module.gallery_db["vehicle"][0]["features"], before
    )
    assert track(client, [1, 0, 0])["global_id"] == first["global_id"]
    assert track(client, [0, 1, 0])["global_id"] != first["global_id"]


def test_threshold_update_changes_match_decision(client):
    first = track(client, [1, 0])
    assert client.patch("/config", json={"vehicle": 0.99}).status_code == 200
    assert track(client, [0.8, 0.6])["global_id"] != first["global_id"]


def test_rejects_invalid_class(client):
    response = client.post(
        "/register_track",
        json={"camera_id": "one", "yolo_class": 99, "features": [1, 0]},
    )
    assert response.status_code == 400
    assert client.get("/stats").json()["total_sightings"] == 0
