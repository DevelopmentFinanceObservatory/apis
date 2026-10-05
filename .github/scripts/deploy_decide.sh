#!/usr/bin/env bash
# Decides whether the resolved target commit needs a deploy, by comparing it
# against the commit Cloud Run is currently serving. Runs inside a checkout
# with full history (deploy.yml's `deploy` job checks that out with
# fetch-depth: 0 before this step), after gcloud is authenticated against the
# GCP project the service lives in.
#
# Env:
#   SERVICE      Cloud Run service name (vars.SERVICE_APIS) -- required
#   REGION       Cloud Run region (vars.REGION) -- required
#   PROJECT_ID   GCP project id (vars.PROJECT_ID) -- required
#   TARGET       the commit deploy.yml resolved as the deploy target --
#                required
#   FORCE        "true" on workflow_dispatch (deploy.yml passes
#                github.event_name == 'workflow_dispatch'); forces
#                deploy=true once the live image is confirmed readable,
#                skipping the live-vs-target comparison, since dispatch
#                exists to force a redeploy of exactly the dispatched
#                commit. Unset or any other value runs the normal
#                comparison. Optional, defaults to false.
#
# Writes `deploy=true` or `deploy=false` to $GITHUB_OUTPUT (or stdout, when
# $GITHUB_OUTPUT is unset, so this runs the same way under a test). Emits
# ::notice::/::warning::/::error:: annotations for the cases that need one.
# Exits non-zero only when the live image can't be read at all: every other
# outcome is a legitimate decision, not a failure.

set -euo pipefail

: "${SERVICE:?SERVICE is required}"
: "${REGION:?REGION is required}"
: "${PROJECT_ID:?PROJECT_ID is required}"
: "${TARGET:?TARGET is required}"
FORCE="${FORCE:-false}"

image="$(gcloud run services describe "$SERVICE" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --format='value(spec.template.spec.containers[0].image)')" || {
  echo "::error::gcloud run services describe failed for service '$SERVICE'."
  exit 1
}

if [ -z "$image" ]; then
  echo "::error::gcloud run services describe returned no image for service '$SERVICE'."
  exit 1
fi

live="${image##*:}"
deploy=

if [ "$FORCE" = "true" ]; then
  echo "::notice::deploy forced by workflow_dispatch: deploying $TARGET regardless of the live image $live."
  deploy=true
elif [ "$live" = "$TARGET" ]; then
  deploy=false
elif git cat-file -e "${live}^{commit}" 2>/dev/null; then
  if git diff --quiet "$live" "$TARGET" -- \
    src packages pyproject.toml uv.lock .python-version Dockerfile \
    .github/workflows/deploy.yml .github/workflows/deploy-dev.yml .github/workflows/deploy-prod.yml
  then
    deploy=false
  else
    deploy=true
  fi
else
  echo "::warning::live image tag '$live' on '$SERVICE' is not a commit this checkout knows, so its inputs can't be compared against target $TARGET. Deploying."
  deploy=true
fi

echo "target=$TARGET live=$live deploy=$deploy"

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "deploy=$deploy" >>"$GITHUB_OUTPUT"
else
  echo "deploy=$deploy"
fi
