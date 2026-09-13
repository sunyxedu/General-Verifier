#!/bin/bash
# Public command transport only; evaluator authority stays in the sidecar.
exec /usr/bin/python3.11 -I /opt/gv-transport/submission.py "$@"
