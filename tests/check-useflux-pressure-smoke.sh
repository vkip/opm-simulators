#!/usr/bin/env bash
set -euo pipefail

flow_bin="$1"
consumer_deck="$2"

if [[ ! -f FLUX_DUMP_PRESSURE_SMOKE.FLUX ]]; then
    echo "Expected FLUX_DUMP_PRESSURE_SMOKE.FLUX in test result directory" >&2
    exit 1
fi

cp "$consumer_deck" FLUX_USE_PRESSURE_SMOKE.DATA
"$flow_bin" FLUX_USE_PRESSURE_SMOKE.DATA > useflux_pressure_smoke.log 2>&1

grep -q "End of simulation" useflux_pressure_smoke.log

# A day past the data: by default a warning, and an error with the second
# item of USEFLUX set to NO.
perl -0pe 's/^TSTEP\n  1 \//TSTEP\n  1 1 \//m' "$consumer_deck" > FLUX_USE_BEYOND_SMOKE.DATA
"$flow_bin" FLUX_USE_BEYOND_SMOKE.DATA > useflux_beyond_smoke.log 2>&1
grep -q "End of simulation" useflux_beyond_smoke.log
grep -q "past the last record of the FLUX file" FLUX_USE_BEYOND_SMOKE.PRT

perl -0pe 's/^(USEFLUX\n  FLUX_DUMP_PRESSURE_SMOKE) \//$1 NO \//m' FLUX_USE_BEYOND_SMOKE.DATA > FLUX_USE_BEYOND_NO_SMOKE.DATA
if "$flow_bin" FLUX_USE_BEYOND_NO_SMOKE.DATA > useflux_beyond_no_smoke.log 2>&1; then
    echo "Expected USEFLUX with NO to stop a run going past the data" >&2
    exit 1
fi
grep -q "the second item of USEFLUX is NO" useflux_beyond_no_smoke.log
