#!/usr/bin/env bash
# Points the scheduled origin catalogue job at the image the apis service
# serves, when it runs another one. A job writes only its own image's format
# root, and the serving revision reads its own image's root, so a job on any
# other image leaves the served catalogue unrefreshed. deploy.yml runs this
# right after the decide step, before anything else moves, so a deploy that
# stopped between the traffic move and the job's update, or whose update
# failed, is repaired before this run builds or stops.
#
# The serving image comes from serving_image.sh: the image of every revision
# that `status.traffic` gives more than 0 percent. Traffic split across
# revisions running different images has no one serving image, and the script
# fails with `::error::` and moves nothing.
#
# Env:
#   SERVICE      Cloud Run service name (vars.SERVICE_APIS) -- required
#   JOB          scheduled origin catalogue job name (vars.JOB_ORIGIN_CATALOGUE)
#                -- required
#   REGION       Cloud Run region (vars.REGION) -- required
#   PROJECT_ID   GCP project id (vars.PROJECT_ID) -- required
#
# Exits non-zero when the service, a serving revision or the job can't be
# read, when traffic splits across images, or when the update fails.

set -euo pipefail

: "${SERVICE:?SERVICE is required}"
: "${JOB:?JOB is required}"
: "${REGION:?REGION is required}"
: "${PROJECT_ID:?PROJECT_ID is required}"

serving="$("$(dirname "$0")/serving_image.sh")"

job_image="$(gcloud run jobs describe "$JOB" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --format='value(spec.template.spec.template.spec.containers[0].image)')" || {
  echo "::error::gcloud run jobs describe failed for job '$JOB'."
  exit 1
}

if [ "$job_image" = "$serving" ]; then
  echo "Job '$JOB' runs the serving image $serving."
  exit 0
fi

echo "::notice::Job '$JOB' runs ${job_image:-no image}, not the serving image $serving. Pointing it at the serving image."
gcloud run jobs update "$JOB" \
  --image="$serving" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --quiet
