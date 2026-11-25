#!/usr/bin/env bash
# Prints the image the apis service serves: the image of every revision that
# `status.traffic` gives more than 0 percent, which Cloud Run records by digest
# (`<repository>/dfo-apis@sha256:<digest>`). deploy_decide.sh and
# deploy_reconcile.sh compare it with the scheduled origin catalogue job's
# image, which the deploy sets to the same digest.
#
# Fails with `::error::` when the service or a serving revision can't be read,
# when no revision takes traffic, and when the revisions taking traffic run
# different images, since no one image then serves.
#
# Env:
#   SERVICE      Cloud Run service name (vars.SERVICE_APIS) -- required
#   REGION       Cloud Run region (vars.REGION) -- required
#   PROJECT_ID   GCP project id (vars.PROJECT_ID) -- required
#
# Writes the image, and nothing else, to stdout. Messages go to stderr.

set -euo pipefail

: "${SERVICE:?SERVICE is required}"
: "${REGION:?REGION is required}"
: "${PROJECT_ID:?PROJECT_ID is required}"

described="$(gcloud run services describe "$SERVICE" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --format=json)" || {
  echo "::error::gcloud run services describe failed for service '$SERVICE'." >&2
  exit 1
}

# One revision name per line, for each traffic entry above 0 percent, or
# `-` for an entry that names none.
revisions="$(python3 -c '
import json, sys
traffic = (json.loads(sys.argv[1]).get("status") or {}).get("traffic") or []
for entry in traffic:
    if (entry.get("percent") or 0) > 0:
        print(entry.get("revisionName") or "-")
' "$described" 2>/dev/null)" || revisions=""

if [ -z "$revisions" ]; then
  echo "::error::No revision of '$SERVICE' takes traffic in status.traffic." >&2
  exit 1
fi

serving=""
counted=0
while read -r revision; do
  counted=$((counted + 1))
  if [ "$revision" = - ]; then
    echo "::error::A traffic entry of '$SERVICE' above 0 percent names no revision." >&2
    exit 1
  fi
  image="$(gcloud run revisions describe "$revision" \
    --region="$REGION" \
    --project="$PROJECT_ID" \
    --format='value(spec.containers[0].image)')" || image=""
  if [ -z "$image" ]; then
    echo "::error::gcloud run revisions describe returned no image for revision '$revision'." >&2
    exit 1
  fi
  if [ -n "$serving" ] && [ "$image" != "$serving" ]; then
    echo "::error::Traffic on '$SERVICE' is split across revisions running different images ($serving and $image), so no one image serves." >&2
    exit 1
  fi
  serving="$image"
done <<<"$revisions"

[ "$counted" -gt 0 ] || {
  echo "::error::read no serving revision of '$SERVICE'." >&2
  exit 1
}
echo "$serving"
