# list all recipes
default:
  just --list

# do a dev install
dev:
  pip install -e '.[dev]'

# run code checks
check:
  #!/usr/bin/env bash

  error=0
  trap error=1 ERR

  echo
  (set -x; ruff . )

  echo
  ( set -x; black --check . )

  echo
  ( set -x; mypy . )

  test $error = 0

# fix code issues
fix:
  black .
  ruff --fix .
