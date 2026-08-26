"""Bearer-token authentication backed by a root-readable JSON credential."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class User:
    username: str
    token_sha256: str
    admin: bool = False


class UserStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._mtime_ns = -1
        self._users: tuple[User, ...] = ()

    def _reload(self) -> None:
        stat = self.path.stat()
        if stat.st_mtime_ns == self._mtime_ns:
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("users credential must be a non-empty JSON array")
        users: list[User] = []
        usernames: set[str] = set()
        digests: set[str] = set()
        for record in payload:
            username = str(record["username"]).strip()
            digest = str(record["token_sha256"]).lower()
            if not username or len(username) > 32:
                raise ValueError("usernames must contain 1-32 characters")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"invalid token digest for {username}")
            if username in usernames or digest in digests:
                raise ValueError("usernames and token digests must be unique")
            usernames.add(username)
            digests.add(digest)
            users.append(User(username, digest, bool(record.get("admin", False))))
        self._users = tuple(users)
        self._mtime_ns = stat.st_mtime_ns

    def authenticate(self, token: str) -> User | None:
        self._reload()
        if not token or len(token) > 512:
            return None
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        for user in self._users:
            if hmac.compare_digest(digest, user.token_sha256):
                return user
        return None
