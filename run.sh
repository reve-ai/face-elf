#!/usr/bin/bash
set -e -u -o pipefail
source venv/bin/activate
W=640
H=480
python3 -m src.detect.main --camera 0 --width $W --height $H --conf-threshold 0.6
