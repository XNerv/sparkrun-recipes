#!/usr/bin/env bash

# Entry point for the effort-alias mod.
# The sparkrun mod contract requires run.sh at the mod directory root.
# Logic lives in effort_alias.py alongside it; exec passes through exit code.
set -euo pipefail
exec python3 "$(dirname "$0")/effort_alias.py"