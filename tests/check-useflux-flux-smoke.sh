#!/usr/bin/env bash
set -euo pipefail

flow_bin="$1"
consumer_deck="$2"

if [[ ! -f FLUX_DUMP_USEFLUX_FLUX_SMOKE.FLUX ]]; then
    echo "Expected FLUX_DUMP_USEFLUX_FLUX_SMOKE.FLUX in test result directory" >&2
    exit 1
fi

cp "$consumer_deck" FLUX_USE_FLUX_SMOKE.DATA
"$flow_bin" FLUX_USE_FLUX_SMOKE.DATA > useflux_flux_smoke.log 2>&1

grep -q "Total number of active cells: 50" useflux_flux_smoke.log
grep -q "End of simulation" useflux_flux_smoke.log

# A file without pressure data cannot drive a FLUXTYPE PRESSURE run.
perl -0pe 's/^FLUXTYPE\n  FLUX \//FLUXTYPE\n  PRESSURE \//m' "$consumer_deck" > FLUX_USE_WRONG_TYPE_SMOKE.DATA
if "$flow_bin" FLUX_USE_WRONG_TYPE_SMOKE.DATA > useflux_wrong_type_smoke.log 2>&1; then
    echo "Expected a FLUXTYPE PRESSURE run on a FLUX-only file to fail" >&2
    exit 1
fi
grep -q "FLUXTYPE PRESSURE needs pressure boundary data" useflux_wrong_type_smoke.log
