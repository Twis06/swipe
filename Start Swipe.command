#!/bin/zsh
cd -- "$(dirname -- "$0")"
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m swipe
fi
exec python3 -m swipe
