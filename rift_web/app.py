"""FastAPI application for one-step, manually curated audio tasks."""

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, Response

from rift_web.audio import ALLOWED_SUFFIXES, probe_audio, sha256
from rift_web.auth import User, UserStore
from rift_web.config import Settings
from rift_web.database import JOB_KINDS, Database, now_iso

KIND_LABELS = {
    "background": "去背景",
    "deharmony": "去和声",
    "dereverb": "去混响",
    "rift": "RIFT",
}

RIFT_PARAMS = {
    "key_shift": (-24, 24, int),
    "steps": (2, 128, int),
    "ds": (0.0, 2.0, float),
    "spk": (0.0, 2.0, float),
    "cfg_rescale": (0.0, 2.0, float),
    "robust_f0": (0, 2, int),
    "seed": (0, 2**31 - 1, int),
}


def parse_rift_params(values: dict[str, str], kind: str) -> str:
    if kind != "rift":
        return "{}"
    parsed: dict[str, int | float] = {}
    for name, (minimum, maximum, converter) in RIFT_PARAMS.items():
        raw = values.get(name, "")
        if not raw:
            continue
        try:
            value = converter(raw)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"RIFT 参数无效：{name}") from None
        if not minimum <= value <= maximum:
            raise HTTPException(status_code=422, detail=f"RIFT 参数超出范围：{name}")
        parsed[name] = value
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    settings.state_directory.mkdir(parents=True, exist_ok=True)
    settings.jobs_directory.mkdir(parents=True, exist_ok=True)
    database = Database(settings.database_path)
    database.initialize()
    users = UserStore(settings.users_file)
    static_index = Path(__file__).parent / "static" / "index.html"

    app = FastAPI(title="RIFT Queue", docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.database = database

    def request_token(request: Request, authorization: str | None) -> str:
        if authorization and authorization.startswith("Bearer "):
            return authorization.removeprefix("Bearer ").strip()
        return request.cookies.get("rift_token", "")

    def current_user(
        request: Request, authorization: str | None = Header(default=None)
    ) -> User:
        token = request_token(request, authorization)
        if not token:
            raise HTTPException(status_code=401, detail="需要口令")
        user = users.authenticate(token)
        if user is None:
            raise HTTPException(status_code=401, detail="口令无效")
        return user

    def optional_user(
        request: Request, authorization: str | None = Header(default=None)
    ) -> User:
        token = request_token(request, authorization)
        return users.authenticate(token) or User("", "", False)

    def serialize_job(
        row: object,
        user: User,
        *,
        detail: bool = False,
        queue_position: int | None = None,
    ) -> dict:
        record = dict(row)
        owned = user.admin or record["username"] == user.username
        result = {
            "id": record["id"],
            "username": record["username"],
            "title": record["title"],
            "kind": record["kind"],
            "kind_label": KIND_LABELS[record["kind"]],
            "status": record["status"],
            "created_at": record["created_at"],
            "started_at": record["started_at"],
            "finished_at": record["finished_at"],
            "public_error": record["public_error"],
            "queue_position": queue_position,
            "owned": owned,
        }
        if record["kind"] == "rift":
            result["params"] = json.loads(record.get("params_json") or "{}")
        if detail and owned:
            result.update(
                {
                    "original_name": record["original_name"],
                    "input_sha256": record["input_sha256"],
                    "input_bytes": record["input_bytes"],
                    "audio": json.loads(record["audio_json"]),
                    "files": [
                        {
                            "name": file["name"],
                            "sha256": file["sha256"],
                            "bytes": file["bytes"],
                            "media_type": file["media_type"],
                            "url": f"/api/jobs/{record['id']}/files/{file['name']}",
                        }
                        for file in database.list_files(record["id"])
                    ],
                }
            )
        return result

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_index, media_type="text/html")

    @app.get("/health")
    def health() -> dict[str, object]:
        database.initialize()
        stat = os.statvfs(settings.state_directory)
        return {
            "status": "ok",
            "free_bytes": stat.f_bavail * stat.f_frsize,
            "dispatcher_heartbeat": database.metadata().get("dispatcher_heartbeat"),
        }

    @app.get("/api/me")
    def me(user: Annotated[User, Depends(current_user)]) -> dict[str, object]:
        return {"username": user.username, "admin": user.admin}

    @app.post("/api/session")
    def create_session(
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
    ) -> JSONResponse:
        token = request_token(request, authorization)
        user = users.authenticate(token)
        if user is None:
            raise HTTPException(status_code=401, detail="口令无效")
        response = JSONResponse({"username": user.username, "admin": user.admin})
        response.set_cookie(
            "rift_token",
            token,
            max_age=30 * 24 * 60 * 60,
            secure=True,
            httponly=True,
            samesite="strict",
            path="/",
        )
        return response

    @app.delete("/api/session", status_code=204)
    def delete_session() -> Response:
        response = Response(status_code=204)
        response.delete_cookie("rift_token", path="/")
        return response

    @app.get("/api/jobs")
    def list_jobs(user: Annotated[User, Depends(optional_user)]) -> list[dict]:
        rows = database.list_jobs()
        queued = sorted(
            (row for row in rows if row["status"] == "queued"),
            key=lambda row: (row["created_at"], row["id"]),
        )
        positions = {row["id"]: index for index, row in enumerate(queued, start=1)}
        return [
            serialize_job(row, user, queue_position=positions.get(row["id"]))
            for row in rows
        ]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, user: Annotated[User, Depends(current_user)]) -> dict:
        row = database.get_job(job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return serialize_job(row, user, detail=True)

    @app.post("/api/jobs", status_code=201)
    async def create_job(
        title: Annotated[str, Form()],
        kind: Annotated[str, Form()],
        upload: Annotated[UploadFile, File()],
        request: Request,
        user: Annotated[User, Depends(current_user)],
    ) -> dict:
        title = title.strip()
        if not 1 <= len(title) <= 100:
            raise HTTPException(status_code=422, detail="任务名必须为 1-100 个字符")
        if kind not in JOB_KINDS:
            raise HTTPException(status_code=422, detail="不支持该处理类型")
        params_json = parse_rift_params(dict(await request.form()), kind)
        original_name = Path(upload.filename or "audio").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=422, detail="不支持该文件扩展名")

        job_id = uuid.uuid4().hex
        job_directory = settings.jobs_directory / job_id
        staging = job_directory / ".uploading"
        input_directory = job_directory / "input"
        input_path = input_directory / f"source{suffix}"
        input_directory.mkdir(parents=True, mode=0o700)
        total = 0
        try:
            with staging.open("xb") as handle:
                while block := await upload.read(1024 * 1024):
                    total += len(block)
                    if total > settings.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="文件超过上传限制")
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
            if total == 0:
                raise HTTPException(status_code=422, detail="上传文件为空")
            try:
                audio = probe_audio(staging)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                raise HTTPException(
                    status_code=422, detail=f"无法验证音频：{error}"
                ) from None
            staging.replace(input_path)
            record = {
                "id": job_id,
                "username": user.username,
                "title": title,
                "kind": kind,
                "original_name": original_name,
                "input_path": str(input_path),
                "input_sha256": sha256(input_path),
                "input_bytes": total,
                "audio_json": json.dumps(audio, ensure_ascii=False),
                "params_json": params_json,
                "created_at": now_iso(),
            }
            database.create_job(record)
        except BaseException:
            shutil.rmtree(job_directory, ignore_errors=True)
            raise
        finally:
            await upload.close()
        row = database.get_job(job_id)
        assert row is not None
        return serialize_job(row, user, detail=True)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(
        job_id: str, user: Annotated[User, Depends(current_user)]
    ) -> dict[str, bool]:
        if not database.cancel(job_id, user.username, user.admin):
            raise HTTPException(status_code=409, detail="只能取消自己的等待中任务")
        return {"cancelled": True}

    @app.get("/api/jobs/{job_id}/files/{name}")
    def download_file(
        job_id: str, name: str, user: Annotated[User, Depends(current_user)]
    ) -> FileResponse:
        job = database.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        if not user.admin and job["username"] != user.username:
            raise HTTPException(status_code=403, detail="只能访问自己的任务文件")
        record = database.get_file(job_id, name)
        if record is None:
            raise HTTPException(status_code=404, detail="文件不存在")
        path = Path(record["path"])
        if not path.is_file():
            raise HTTPException(status_code=410, detail="文件已经过期删除")
        return FileResponse(
            path,
            media_type=record["media_type"],
            filename=record["name"],
        )

    return app
