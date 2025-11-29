#!/usr/bin/bash
set -e -u -o pipefail
source venv/bin/activate
PYTHONPATH=src python3 -m detect.normalize_tool "$@"
