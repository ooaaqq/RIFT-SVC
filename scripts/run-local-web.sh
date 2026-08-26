#!/usr/bin/env bash

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state_directory="${RIFT_LOCAL_WEB_STATE_DIRECTORY:-$root/var/rift-web-local}"
users_file="$state_directory/users.json"

mkdir -p "$state_directory"
chmod 0700 "$state_directory"

token_hash() {
  printf %s "$1" | sha256sum | cut -d ' ' -f1
}

jq -n \
  --arg admin "$(token_hash local-admin)" \
  --arg friend "$(token_hash local-friend)" \
  '[
    {username:"本地管理员",token_sha256:$admin,admin:true},
    {username:"本地好友",token_sha256:$friend,admin:false}
  ]' > "$users_file"
chmod 0600 "$users_file"

export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export RIFT_WEB_SOURCE_ROOT="$root"
export RIFT_WEB_STATE_DIRECTORY="$state_directory"
export RIFT_WEB_USERS_FILE="$users_file"
export RIFT_WEB_LISTEN_HOST="${RIFT_WEB_LISTEN_HOST:-127.0.0.1}"
export RIFT_WEB_LISTEN_PORT="${RIFT_WEB_LISTEN_PORT:-8767}"

printf 'Local RIFT web preview: http://%s:%s\n' "$RIFT_WEB_LISTEN_HOST" "$RIFT_WEB_LISTEN_PORT"
printf 'Local tokens: local-admin / local-friend\n'
printf 'State: %s\n' "$state_directory"
exec uv run --extra web uvicorn rift_web.app:create_app \
  --factory \
  --host "$RIFT_WEB_LISTEN_HOST" \
  --port "$RIFT_WEB_LISTEN_PORT" \
  --reload
