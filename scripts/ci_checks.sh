#!/bin/bash

set -e

echo 'running flake8'
flake8 .

echo 'running isort'
isort . --check

echo 'running black'
black . --check --line-length=128

echo 'running mypy'
mypy .
