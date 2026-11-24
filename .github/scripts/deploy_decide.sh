#!/usr/bin/env bash
# Decides what a deploy of the resolved target commit does, from the revision
# that serves the apis service's traffic. Runs inside a checkout with full
# history (deploy.yml's `deploy` job checks that out with fetch-depth: 0 before
# this step), after gcloud is authenticated against the GCP project the service
# lives in.
#
# A revision records its image by digest, and the build pushes an image index
# whose digest differs from the one Cloud Run records, so a revision's commit
# is read from the service template. The template describes the latest created
# revision, whose image tag is its commit. The script reads one
# `gcloud run services describe` and classifies the service:
#
#   serving latest   the revision with 100 percent of traffic is the latest
#                    created revision, so the template's commit serves. A
#                    target equal to it, or one whose diff from it touches
#                    none of the watched paths, needs nothing; any other
#                    target deploys.
#   staged           traffic serves an older revision and a revision that
#                    serves nothing carries the `candidate` tag: a deploy
#                    staged it and stopped before traffic moved, leaving the
#                    traffic pinned by name. When the tagged revision is the
#                    latest created one, the target's, and ready, the deploy
#                    releases it (probe, move traffic, settle, backfill)
#                    without building. Otherwise the target deploys, since the
#                    serving commit is unknown, and its traffic move ends the
#                    pin, so a revision created after the stop (an `infra`
#                    apply) does not stay without traffic.
#   held             traffic serves an older revision and no other revision
#                    carries the `candidate` tag, or traffic is split: someone
#                    moved traffic by hand. A target with the image
#                    inputs of the latest created revision would undo that, so
#                    the script refuses it unless OVERRIDE is set. A target
#                    with other inputs deploys.
#
# Env:
#   SERVICE      Cloud Run service name (vars.SERVICE_APIS) -- required
#   REGION       Cloud Run region (vars.REGION) -- required
#   PROJECT_ID   GCP project id (vars.PROJECT_ID) -- required
#   TARGET       the commit deploy.yml resolved as the deploy target --
#                required
#   FORCE        "true" on workflow_dispatch: deploys the target even when it
#                already serves or has the serving commit's inputs, for
#                example to re-run the migration jobs. A held service still
#                refuses without OVERRIDE. Optional, defaults to false.
#   OVERRIDE     "true" deploys a held service's target anyway, moving the
#                traffic someone set by hand. Optional, defaults to false.
#
# Writes to $GITHUB_OUTPUT (or stdout, when $GITHUB_OUTPUT is unset, so this
# runs the same way under a test):
#
#   deploy=true|false    build the image, run the jobs, stage a new revision
#   release=true|false   probe a revision, move traffic to it, settle, backfill
#   revision=<name>      the staged revision to release without building;
#                        empty when deploy=true, where the staging step names
#                        the new revision
#
# Exits non-zero when the service can't be read, and when it refuses a held
# service.

set -euo pipefail

: "${SERVICE:?SERVICE is required}"
: "${REGION:?REGION is required}"
: "${PROJECT_ID:?PROJECT_ID is required}"
: "${TARGET:?TARGET is required}"
FORCE="${FORCE:-false}"
OVERRIDE="${OVERRIDE:-false}"

described="$(gcloud run services describe "$SERVICE" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --format=json)" || {
  echo "::error::gcloud run services describe failed for service '$SERVICE'."
  exit 1
}

# Five fields: the template's image, the latest created and latest ready
# revisions, the revision with 100 percent of traffic, and the revision tagged
# `candidate`. A missing field prints as `-`.
fields="$(python3 -c '
import json, sys
service = json.loads(sys.argv[1])
status = service.get("status") or {}
traffic = status.get("traffic") or []
containers = ((service.get("spec") or {}).get("template") or {}).get("spec", {}).get("containers") or [{}]
serving = [e.get("revisionName") for e in traffic if e.get("percent") == 100]
candidate = [e.get("revisionName") for e in traffic if e.get("tag") == "candidate"]
fields = [
    containers[0].get("image"),
    status.get("latestCreatedRevisionName"),
    status.get("latestReadyRevisionName"),
    serving[0] if serving else None,
    candidate[0] if candidate else None,
]
print(" ".join(field or "-" for field in fields))
' "$described" 2>/dev/null)" || fields=""
read -r image latest ready serving candidate <<<"${fields:-- - - - -}"

if [ "$image" = - ] || [ "$latest" = - ]; then
  echo "::error::gcloud run services describe returned no image or latest revision for service '$SERVICE'."
  exit 1
fi

# The latest created revision's commit: its image tag.
commit="${image##*:}"

# Prints `same` when the target has the image inputs of the given commit,
# `differs` when it does not, and `unknown` when the checkout doesn't know the
# commit.
inputs_of() {
  if [ "$1" = "$TARGET" ]; then
    echo same
  elif ! git cat-file -e "${1}^{commit}" 2>/dev/null; then
    echo unknown
  elif git diff --quiet "$1" "$TARGET" -- \
    src packages pyproject.toml uv.lock .python-version Dockerfile \
    .github/workflows/deploy.yml .github/workflows/deploy-dev.yml .github/workflows/deploy-prod.yml; then
    echo same
  else
    echo differs
  fi
}

deploy=false
release=false
revision=

if [ "$serving" = "$latest" ]; then
  state="serving-latest"
  inputs="$(inputs_of "$commit")"
  if [ "$FORCE" = true ]; then
    echo "::notice::deploy forced by workflow_dispatch: deploying $TARGET over the serving commit $commit."
    deploy=true
  elif [ "$inputs" = unknown ]; then
    echo "::warning::the serving image tag '$commit' on '$SERVICE' is not a commit this checkout knows, so its inputs can't be compared against target $TARGET. Deploying."
    deploy=true
  elif [ "$inputs" = differs ]; then
    deploy=true
  fi
elif [ "$serving" != - ] && [ "$candidate" != - ] && [ "$candidate" != "$serving" ]; then
  state=staged
  if [ "$FORCE" != true ] && [ "$candidate" = "$latest" ] && [ "$commit" = "$TARGET" ] && [ "$ready" = "$latest" ]; then
    echo "::notice::revision $latest of $TARGET was staged by an earlier run that stopped before traffic moved; releasing it."
    release=true
    revision="$latest"
  else
    deploy=true
  fi
else
  state=held
  inputs="$(inputs_of "$commit")"
  if [ "$inputs" != differs ] && [ "$OVERRIDE" != true ]; then
    if [ "$serving" = - ]; then
      held="traffic is split across revisions"
    else
      held="revision $serving serves"
    fi
    echo "::error::Traffic on '$SERVICE' was moved by hand: $held, and the latest revision $latest (commit $commit) serves none. Target $TARGET has the image inputs of $commit, so deploying it would undo that move. Dispatch the deploy with override_manual_traffic to deploy it anyway."
    exit 1
  fi
  deploy=true
fi

if [ "$deploy" = true ]; then
  release=true
fi

echo "target=$TARGET state=$state serving=$serving latest=$latest commit=$commit candidate=$candidate deploy=$deploy release=$release revision=${revision:--}"

outputs="$(printf 'deploy=%s\nrelease=%s\nrevision=%s' "$deploy" "$release" "$revision")"
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "$outputs" >>"$GITHUB_OUTPUT"
else
  echo "$outputs"
fi
