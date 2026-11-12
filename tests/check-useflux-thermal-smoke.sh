#!/usr/bin/env bash
set -euo pipefail

# A thermal sector whose cold inflow arrives across an ordinary face and an NNC.
#
# The parent injects 20 C water outside the sector into an 80 C reservoir and
# writes both kinds of boundary data. The sector is then run on that file with
# pressure boundaries and with flux boundaries, and the temperature of its
# three cells is compared with the parent's.
#
# What each tolerance guards against: without the exterior temperature in the
# file the NNC cell ends up 3.2 C off in pressure mode, and without the
# enthalpy of the fixed boundary rates the cells next to the ordinary faces
# end up 6 to 8 C off in flux mode. With both in place the sector stays within
# 1.1 C and 0.4 C of the parent.

flow_bin="$1"
summary_bin="$2"
consumer_deck="$3"

producer_base="TWOWELL_DUMPFLUX_THERMAL_1D"
consumer_base="$(basename "$consumer_deck" .DATA)"

for array in FLXTEMP FLXTHTR; do
    if ! grep -a -q "$array" "${producer_base}.FLUX"; then
        echo "check-useflux-thermal-smoke: ${array} missing from ${producer_base}.FLUX" >&2
        exit 1
    fi
done

run_sector() {
    local mode="$1"
    rm -rf "$mode"
    mkdir -p "$mode"
    cp "${producer_base}.FLUX" "$mode/${consumer_base}.FLUX"
    sed "s/^  PRESSURE \/$/  ${mode} \//" "$consumer_deck" > "$mode/${consumer_base}.DATA"
    (cd "$mode" && "$flow_bin" "${consumer_base}.DATA" --output-dir=. > use.log 2>&1)
    grep -q "End of simulation" "$mode/use.log"
}

max_deviation() {
    local mode="$1" key="$2"
    paste <("$summary_bin" "${producer_base}.SMSPEC" "$key" 2>/dev/null) \
          <("$summary_bin" "$mode/${consumer_base}.SMSPEC" "$key" 2>/dev/null) \
        | awk 'NF == 2 && $1 + 0 == $1 { d = $2 - $1; if (d < 0) d = -d; if (d > m) m = d; n++ }
               END { if (n == 0) exit 1; print m }'
}

failed=0
for spec in PRESSURE:1.5 FLUX:0.75; do
    mode="${spec%%:*}"
    tolerance="${spec##*:}"
    run_sector "$mode"
    for cell in 3,1,1 4,1,1 5,1,1; do
        deviation="$(max_deviation "$mode" "BTEMP:${cell}")" || {
            echo "check-useflux-thermal-smoke: no BTEMP:${cell} in the ${mode} run" >&2
            failed=1
            continue
        }
        echo "${mode} BTEMP:${cell} max deviation ${deviation} C"
        if awk -v d="$deviation" -v t="$tolerance" 'BEGIN { exit !(d > t) }'; then
            echo "check-useflux-thermal-smoke: ${mode} BTEMP:${cell} off by ${deviation} C," \
                 "more than ${tolerance} C" >&2
            failed=1
        fi
    done
done

exit "$failed"
