#!/usr/bin/env sh
set -eu
exec python -m ragqa serve --index "${RAG_INDEX:-artifacts/demo}" --host 0.0.0.0 "$@"
