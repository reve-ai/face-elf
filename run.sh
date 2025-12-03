#!/usr/bin/bash
set -e -u -o pipefail
source venv/bin/activate
PYTHONPATH=src python3 -m detect.main_gui --elf-mode --camera 0 --conf-threshold 0.6
