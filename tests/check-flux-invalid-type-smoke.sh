#!/bin/bash

set -euo pipefail

flow_bin="$1"
deck="$2"

cp "${deck}" FLUX_DUMP_INVALID_TYPE_SMOKE.DATA
if "${flow_bin}" FLUX_DUMP_INVALID_TYPE_SMOKE.DATA > invalid_type_smoke.log 2>&1; then
    echo "Expected a run with FLUXTYPE BANANA to fail" >&2
    exit 1
fi

grep -q "FLUXTYPE must be FLUX, PRESSURE or BOTH" invalid_type_smoke.log
