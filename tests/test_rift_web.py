from __future__ import annotations

import hashlib
import json
import math
import struct
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from rift_web.app import create_app
from rift_web.auth import UserStore
from rift_web.config import Settings
from rift_web.database import Database, now_iso
from rift_web.dispatcher import Dispatcher


def token_record(username: str, token: str, *, admin: bool = False) -> dict:
    return {
        "username": username,
        "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "admin": admin,
    }


def write_users(path: Path) -> None:
    path.write_text(
        json.dumps(
            [token_record("露早", "token-one"), token_record("朋友", "token-two")]
        ),
        encoding="utf-8",
    )


def write_wave(path: Path) -> None:
    sample_rate = 8000
    frames = [
        int(math.sin(2 * math.pi * 440 * index / sample_rate) * 1000)
        for index in range(sample_rate)
    ]
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"".join(struct.pack("<h", frame) for frame in frames))


def settings(tmp_path: Path) -> Settings:
    users = tmp_path / "users.json"
    write_users(users)
    return Settings(
        state_directory=tmp_path / "state",
        users_file=users,
        source_root=Path(__file__).parents[1],
        listen_host="127.0.0.1",
        listen_port=8766,
        max_upload_bytes=2 * 1024 * 1024,
        retention_days=14,
        poll_seconds=1,
    )


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_user_store_authenticates_digest_without_plaintext(tmp_path: Path) -> None:
    path = tmp_path / "users.json"
    write_users(path)
    store = UserStore(path)

    assert store.authenticate("token-one").username == "露早"
    assert store.authenticate("wrong") is None
    assert "token-one" not in path.read_text(encoding="utf-8")


def test_session_uses_secure_http_only_cookie(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path)), base_url="https://rift.test")

    response = client.post("/api/session", headers=auth("token-one"))

    assert response.status_code == 200
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite=strict" in cookie
    assert client.get("/api/me").json()["username"] == "露早"


def test_shared_queue_hides_private_input_metadata(tmp_path: Path) -> None:
    config = settings(tmp_path)
    source = tmp_path / "source.wav"
    write_wave(source)
    client = TestClient(create_app(config))

    with source.open("rb") as audio:
        response = client.post(
            "/api/jobs",
            headers=auth("token-one"),
            data={"title": "测试歌曲", "kind": "rift"},
            files={"upload": ("private-name.wav", audio, "audio/wav")},
        )
    assert response.status_code == 201, response.text
    job_id = response.json()["id"]

    shared = client.get("/api/jobs", headers=auth("token-two")).json()
    assert shared[0]["username"] == "露早"
    assert shared[0]["title"] == "测试歌曲"
    assert "original_name" not in shared[0]
    assert shared[0]["queue_position"] == 1

    detail = client.get(f"/api/jobs/{job_id}", headers=auth("token-two")).json()
    assert "original_name" not in detail
    own_detail = client.get(f"/api/jobs/{job_id}", headers=auth("token-one")).json()
    assert own_detail["original_name"] == "private-name.wav"


def test_selected_model_is_persisted_and_dispatched(tmp_path: Path) -> None:
    config = settings(tmp_path)
    source = tmp_path / "source.wav"
    write_wave(source)
    client = TestClient(create_app(config))

    with source.open("rb") as audio:
        response = client.post(
            "/api/jobs",
            headers=auth("token-one"),
            data={
                "title": "去和声候选",
                "kind": "deharmony",
                "model": "becruily-frazer-karaoke",
            },
            files={"upload": ("source.wav", audio, "audio/wav")},
        )

    assert response.status_code == 201, response.text
    assert response.json()["model"] == "becruily-frazer-karaoke"
    row = Database(config.database_path).get_job(response.json()["id"])
    assert row is not None
    command = Dispatcher(config).command_for(row, tmp_path / "results")
    assert command[command.index("--model") + 1] == "becruily-frazer-karaoke"


def test_rejects_model_from_another_task_type(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    write_wave(source)
    client = TestClient(create_app(settings(tmp_path)))

    with source.open("rb") as audio:
        response = client.post(
            "/api/jobs",
            headers=auth("token-one"),
            data={
                "title": "错误模型",
                "kind": "dereverb",
                "model": "anvuew-karaoke",
            },
            files={"upload": ("source.wav", audio, "audio/wav")},
        )

    assert response.status_code == 422


def test_only_owner_can_download_completed_output(tmp_path: Path) -> None:
    config = settings(tmp_path)
    database = Database(config.database_path)
    database.initialize()
    input_path = config.jobs_directory / "job" / "input" / "source.wav"
    input_path.parent.mkdir(parents=True)
    write_wave(input_path)
    database.create_job(
        {
            "id": "job",
            "username": "露早",
            "title": "任务",
            "kind": "rift",
            "original_name": "source.wav",
            "input_path": str(input_path),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "input_bytes": input_path.stat().st_size,
            "audio_json": "{}",
            "created_at": now_iso(),
        }
    )
    assert database.claim_next()["id"] == "job"
    output = config.jobs_directory / "job" / "output" / "converted.wav"
    output.parent.mkdir(parents=True)
    write_wave(output)
    database.complete(
        "job",
        [
            {
                "name": output.name,
                "path": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "bytes": output.stat().st_size,
                "media_type": "audio/wav",
            }
        ],
    )
    client = TestClient(create_app(config))

    assert client.get("/api/jobs/job/input", headers=auth("token-two")).status_code == 403
    input_response = client.get("/api/jobs/job/input", headers=auth("token-one"))
    assert input_response.status_code == 200
    assert input_response.content == input_path.read_bytes()
    assert (
        client.get(
            "/api/jobs/job/files/converted.wav", headers=auth("token-two")
        ).status_code
        == 403
    )
    assert (
        client.get(
            "/api/jobs/job/files/converted.wav", headers=auth("token-one")
        ).status_code
        == 200
    )


def test_queued_job_can_only_be_cancelled_by_owner(tmp_path: Path) -> None:
    config = settings(tmp_path)
    database = Database(config.database_path)
    database.initialize()
    input_path = config.jobs_directory / "job" / "input" / "source.wav"
    input_path.parent.mkdir(parents=True)
    write_wave(input_path)
    database.create_job(
        {
            "id": "job",
            "username": "露早",
            "title": "任务",
            "kind": "dereverb",
            "original_name": "source.wav",
            "input_path": str(input_path),
            "input_sha256": "0" * 64,
            "input_bytes": input_path.stat().st_size,
            "audio_json": "{}",
            "created_at": now_iso(),
        }
    )
    client = TestClient(create_app(config))

    assert (
        client.post("/api/jobs/job/cancel", headers=auth("token-two")).status_code
        == 409
    )
    assert (
        client.post("/api/jobs/job/cancel", headers=auth("token-one")).status_code
        == 200
    )
