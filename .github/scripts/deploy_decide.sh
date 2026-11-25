#!/usr/bin/env bash
# Decides whether the resolved target commit needs a deploy, by asking whether
# the revision serving the apis service runs that commit's code. Runs inside a
# checkout with full history (deploy.yml's `deploy` job checks that out with
# fetch-depth: 0 before this step), after gcloud is authenticated against the
# GCP project the service lives in.
#
# A revision records its image by digest, and the build pushes an image index
# whose digest differs from the one Cloud Run records, so the serving commit is
# read from the service template, which describes the latest created revision.
# That commit is the serving one only in the state a finished deploy leaves:
# `spec.traffic` sends 100 percent to the latest revision, the latest created
# revision is ready, and no revision carries the `candidate` tag.
# Any other state is unknown (a deploy stopped part-way, a revision that never
# started, or traffic moved by hand), and the target deploys.
#
# In the known state, a target equal to the serving commit, or one whose diff
# from it touches none of the watched paths, needs no deploy. A serving tag
# that isn't a commit this checkout knows (a placeholder image, or a
# since-deleted branch) can't be compared, so the target deploys.
#
# Env:
#   SERVICE      Cloud Run service name (vars.SERVICE_APIS) -- required
#   REGION       Cloud Run region (vars.REGION) -- required
#   PROJECT_ID   GCP project id (vars.PROJECT_ID) -- required
#   TARGET       the commit deploy.yml resolved as the deploy target --
#                required
#   FORCE        "true" on workflow_dispatch: deploys whatever the service
#                state, since dispatch exists to force a redeploy of exactly
#                the dispatched commit. Optional, defaults to false.
#
# Writes `deploy=true` or `deploy=false`, and `serving_image=<the template's
# image>`, to $GITHUB_OUTPUT (or stdout, when $GITHUB_OUTPUT is unset, so this
# runs the same way under a test). The serving image is the one a failed
# deploy points the origin catalogue job back at; in an unknown state it is
# the latest created revision's image, which may not take traffic. Exits
# non-zero only when the service can't be read.

set -euo pipefail

: "${SERVICE:?SERVICE is required}"
: "${REGION:?REGION is required}"
: "${PROJECT_ID:?PROJECT_ID is required}"
: "${TARGET:?TARGET is required}"
FORCE="${FORCE:-false}"

described="$(gcloud run services describe "$SERVICE" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --format=json)" || {
  echo "::error::gcloud run services describe failed for service '$SERVICE'."
  exit 1
}

# Two fields: the template's image, and `known` or `unknown` for whether the
# service is in the state a finished deploy leaves. A missing image prints as
# `-`.
fields="$(python3 -c '
import json, sys
service = json.loads(sys.argv[1])
spec = service.get("spec") or {}
status = service.get("status") or {}
containers = ((spec.get("template") or {}).get("spec") or {}).get("containers") or [{}]
traffic = spec.get("traffic") or []
follows = any(e.get("latestRevision") and e.get("percent") == 100 for e in traffic)
tagged = any(e.get("tag") == "candidate" for e in traffic)
latest = status.get("latestCreatedRevisionName")
known = follows and not tagged and latest and latest == status.get("latestReadyRevisionName")
print(containers[0].get("image") or "-", "known" if known else "unknown")
' "$described" 2>/dev/null)" || fields=""
read -r image state <<<"${fields:-- unknown}"

if [ "$image" = - ]; then
  echo "::error::gcloud run services describe returned no image for service '$SERVICE'."
  exit 1
fi

live="${image##*:}"
if [ "$FORCE" = true ]; then
  echo "::notice::deploy forced by workflow_dispatch: deploying $TARGET."
  deploy=true
elif [ "$state" = unknown ]; then
  echo "::notice::'$SERVICE' is not in the state a finished deploy leaves (traffic following a ready latest revision, no candidate tag), so the serving commit is unknown. Deploying $TARGET."
  deploy=true
elif [ "$live" = "$TARGET" ]; then
  deploy=false
elif ! git cat-file -e "${live}^{commit}" 2>/dev/null; then
  echo "::warning::the serving image tag '$live' on '$SERVICE' is not a commit this checkout knows, so its inputs can't be compared against target $TARGET. Deploying."
  deploy=true
elif git diff --quiet "$live" "$TARGET" -- \
  src packages pyproject.toml uv.lock .python-version Dockerfile \
  .github/workflows/deploy.yml .github/workflows/deploy-dev.yml .github/workflows/deploy-prod.yml; then
  deploy=false
else
  deploy=true
fi

echo "target=$TARGET live=$live state=$state deploy=$deploy"

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  printf 'deploy=%s\nserving_image=%s\n' "$deploy" "$image" >>"$GITHUB_OUTPUT"
else
  printf 'deploy=%s\nserving_image=%s\n' "$deploy" "$image"
fi
