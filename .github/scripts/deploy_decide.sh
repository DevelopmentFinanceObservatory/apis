#!/usr/bin/env bash
# Decides what a deploy of the resolved target commit does, from the traffic
# the apis service asks for. Runs inside a checkout with full history
# (deploy.yml's `deploy` job checks that out with fetch-depth: 0 before this
# step), after gcloud is authenticated against the GCP project the service
# lives in.
#
# A revision records its image by digest, and the build pushes an image index
# whose digest differs from the one Cloud Run records, so a revision's commit
# is read from the service template. The template describes the latest created
# revision, whose image tag is its commit.
#
# The script reads one `gcloud run services describe`. `spec.traffic` records
# the traffic asked for: an entry with `latestRevision: true` and 100 percent
# follows the latest ready revision; an entry naming a revision with 100
# percent pins traffic to it. The `candidate` tag marks a release that has not
# finished: the stage step puts it on the revision it creates, and the last
# step of a release removes it. The script classifies the service:
#
#   unfinished      the revision tagged `candidate` serves: a release moved
#                   traffic and stopped before settle, backfill or the tag
#                   removal finished. When that revision is the latest created
#                   one and the target's, the deploy finishes it (traffic
#                   follows the latest revision again, settle, backfill)
#                   without probing; otherwise the target deploys.
#   staged          a revision that serves nothing carries the `candidate`
#                   tag: a deploy staged it and stopped before traffic moved,
#                   leaving traffic pinned by name. When it is the latest
#                   created revision, ready, the target's, and the datasets
#                   maintenance job still runs the target's image, the deploy
#                   releases it (probe, move traffic, settle, backfill)
#                   without building. Otherwise the target deploys, and its
#                   traffic move ends the pin.
#   serving-latest  traffic follows the latest ready revision, or is pinned to
#                   the latest created revision. A target equal to its commit,
#                   or one whose diff from it touches none of the watched
#                   paths, needs nothing, except that a pin is finished like an
#                   unfinished release; any other target deploys. When the
#                   latest created revision never became ready, an older one
#                   serves with an unknown commit, and the target deploys.
#   held            traffic is pinned to an older revision and no revision
#                   carries the `candidate` tag, or traffic is split: someone
#                   moved traffic by hand. A target with the image inputs of
#                   the latest created revision would undo that, so the script
#                   refuses it unless OVERRIDE is set. Any other target
#                   deploys.
#
# Env:
#   SERVICE      Cloud Run service name (vars.SERVICE_APIS) -- required
#   REGION       Cloud Run region (vars.REGION) -- required
#   PROJECT_ID   GCP project id (vars.PROJECT_ID) -- required
#   TARGET       the commit deploy.yml resolved as the deploy target --
#                required
#   MAINTENANCE_JOB
#                the datasets maintenance job (vars.JOB_DATASETS_MAINTENANCE),
#                whose image a staged release must match -- required
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
#   probe=true|false     probe the candidate revision before traffic moves
#   release=true|false   move traffic, settle, backfill, remove the tag
#   revision=<name>      the revision to release or finish without building;
#                        empty when deploy=true, where the stage step names
#                        the new revision
#
# Exits non-zero when the service can't be read, and when it refuses a held
# service.

set -euo pipefail

: "${SERVICE:?SERVICE is required}"
: "${REGION:?REGION is required}"
: "${PROJECT_ID:?PROJECT_ID is required}"
: "${TARGET:?TARGET is required}"
: "${MAINTENANCE_JOB:?MAINTENANCE_JOB is required}"
FORCE="${FORCE:-false}"
OVERRIDE="${OVERRIDE:-false}"

described="$(gcloud run services describe "$SERVICE" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --format=json)" || {
  echo "::error::gcloud run services describe failed for service '$SERVICE'."
  exit 1
}

# Six fields: the template's image, the latest created and latest ready
# revisions, how traffic is asked for (`latest`, `pinned` or `split`), the
# revision a pin names, and the revision tagged `candidate`. A missing field
# prints as `-`.
fields="$(python3 -c '
import json, sys
service = json.loads(sys.argv[1])
spec = service.get("spec") or {}
status = service.get("status") or {}
containers = ((spec.get("template") or {}).get("spec") or {}).get("containers") or [{}]
traffic = spec.get("traffic") or []
whole = [e for e in traffic if e.get("percent") == 100]
if whole and whole[0].get("latestRevision"):
    mode, pinned = "latest", None
elif whole:
    mode, pinned = "pinned", whole[0].get("revisionName")
else:
    mode, pinned = "split", None
candidate = [e.get("revisionName") for e in traffic if e.get("tag") == "candidate"]
fields = [
    containers[0].get("image"),
    status.get("latestCreatedRevisionName"),
    status.get("latestReadyRevisionName"),
    mode,
    pinned,
    candidate[0] if candidate else None,
]
print(" ".join(field or "-" for field in fields))
' "$described" 2>/dev/null)" || fields=""
read -r image latest ready mode pinned candidate <<<"${fields:-- - - - - -}"

if [ "$image" = - ] || [ "$latest" = - ]; then
  echo "::error::gcloud run services describe returned no image or latest revision for service '$SERVICE'."
  exit 1
fi

# The latest created revision's commit: its image tag.
commit="${image##*:}"

case "$mode" in
  latest) serving="$ready" ;;
  pinned) serving="$pinned" ;;
  *) serving=- ;;
esac

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

# The commit the datasets maintenance job runs: its image tag.
job_commit() {
  local job_image
  job_image="$(gcloud run jobs describe "$MAINTENANCE_JOB" \
    --region="$REGION" \
    --project="$PROJECT_ID" \
    --format='value(spec.template.spec.template.spec.containers[0].image)')" || job_image=""
  echo "${job_image##*:}"
}

deploy=false
probe=false
release=false
revision=

if [ "$candidate" != - ] && [ "$candidate" = "$serving" ]; then
  state=unfinished
  if [ "$FORCE" != true ] && [ "$candidate" = "$latest" ] && [ "$commit" = "$TARGET" ]; then
    echo "::notice::revision $candidate of $TARGET serves, and its release stopped before it finished; finishing it."
    release=true
    revision="$candidate"
  else
    deploy=true
  fi
elif [ "$candidate" != - ]; then
  state=staged
  if [ "$FORCE" != true ] && [ "$candidate" = "$latest" ] && [ "$ready" = "$latest" ] &&
    [ "$commit" = "$TARGET" ] && [ "$(job_commit)" = "$TARGET" ]; then
    echo "::notice::revision $candidate of $TARGET was staged by an earlier run that stopped before traffic moved; releasing it."
    probe=true
    release=true
    revision="$candidate"
  else
    deploy=true
  fi
elif [ "$mode" = latest ] && [ "$ready" != "$latest" ]; then
  state="latest-not-ready"
  echo "::notice::the latest revision $latest of '$SERVICE' never became ready, so $ready serves with a commit this script can't read. Deploying."
  deploy=true
elif [ "$serving" = "$latest" ]; then
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
  elif [ "$mode" = pinned ]; then
    echo "::notice::traffic is pinned by name to the latest revision $latest, as a release that stopped after moving traffic leaves it; finishing that release."
    release=true
    revision="$latest"
  fi
else
  state=held
  inputs="$(inputs_of "$commit")"
  if [ "$inputs" = same ] && [ "$OVERRIDE" != true ]; then
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
  probe=true
  release=true
fi

echo "target=$TARGET state=$state mode=$mode serving=$serving latest=$latest ready=$ready commit=$commit candidate=$candidate deploy=$deploy probe=$probe release=$release revision=${revision:--}"

outputs="$(printf 'deploy=%s\nprobe=%s\nrelease=%s\nrevision=%s' "$deploy" "$probe" "$release" "$revision")"
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "$outputs" >>"$GITHUB_OUTPUT"
else
  echo "$outputs"
fi
