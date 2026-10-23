#!/usr/bin/env sh
set -eu
exec python -m ragqa ui --index "${RAG_INDEX:-artifacts/demo-v2}" "$@"
