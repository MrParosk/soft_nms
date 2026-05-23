#!/bin/bash

set -e

echo 'running ruff lint'
ruff check .

echo 'running ruff format'
ruff format . --check

echo 'running mypy'
mypy .
