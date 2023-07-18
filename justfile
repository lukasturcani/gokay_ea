# list all recipes
default:
  just --list

# do a dev install
dev:
  pip install -e '.[dev]'
