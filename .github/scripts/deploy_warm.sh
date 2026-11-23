#!/usr/bin/env bash
# Calls the apis service's warm paths once the new revision serves, so the
# builds that run beside serving finish before readers arrive. The Search and
# Skills surfaces build the skills index, and the Graph surface builds the
# schema document, after the lifespan returns. Cloud Run
# throttles the CPU of an instance with no request in flight, so a revision
# nobody calls stalls those builds until the first readers arrive and pay for
# them. Each warm path's handler waits on one of those builds:
#
#   /api/skills                        the skills index's first build
#                                      (`skills_api.deps.built_index`, up to
#                                      30 s, then 503)
#   /api/graph/schema                  the schema document (up to 10 s, then
#                                      503 `schema_building`)
#
# A request keeps its instance's CPU allocated, so the builds run while the
# script waits, and a 503 the script retries means the build runs on.
#
# Runs after gcloud is authenticated as the deployer service account. The
# deployer mints an ID token for the service URL through the IAM Credentials
# generateIdToken call on itself, which needs roles/run.invoker on the service
# and roles/iam.serviceAccountOpenIdTokenCreator on itself (both granted in
# the infra repository's apis stack, main.tf).
#
# Env:
#   SERVICE      Cloud Run service name (vars.SERVICE_APIS) -- required
#   REGION       Cloud Run region (vars.REGION) -- required
#   PROJECT_ID   GCP project id (vars.PROJECT_ID) -- required
#   DEPLOYER     email of the deployer service account (secrets.WIF_SA) --
#                required
#   WARM_DEADLINE_SECONDS
#                how long every path together may take, retries included.
#                Optional, defaults to 600.
#   IAM_CREDENTIALS_BASE
#                IAM Credentials API root. Optional; tests point it at a
#                local server.
#
# Writes `status=<outcome>` to $GITHUB_OUTPUT (or stdout, when $GITHUB_OUTPUT
# is unset, so this runs the same way under a test):
#
#   warm         every path answered 200
#   slow         a path still answered 502, 503, 504 or 429, or no answer, at the
#                deadline; a warning, since the builds run on for the next
#                request
#   forbidden    the token mint or the service answered 401 or 403, as it
#                does until the infra grants apply; a warning. Probing stops
#                at the first 401 or 403, and the status is `error` instead
#                when an earlier path answered an error.
#   unavailable  the service URL or the token could not be read for another
#                reason; a warning
#   error        the service answered a path with any other status; an
#                error annotation per path
#
# Exits 0 for every outcome, so the deploy steps after this one still run.
# deploy.yml's last step fails the job on `error`. Exits non-zero only on a
# missing required variable.

set -euo pipefail

: "${SERVICE:?SERVICE is required}"
: "${REGION:?REGION is required}"
: "${PROJECT_ID:?PROJECT_ID is required}"
: "${DEPLOYER:?DEPLOYER is required}"
deadline_seconds="${WARM_DEADLINE_SECONDS:-600}"
iam_base="${IAM_CREDENTIALS_BASE:-https://iamcredentials.googleapis.com}"

paths=(/api/skills /api/graph/schema)

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

finish() {
  echo "warm status=$1"
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "status=$1" >>"$GITHUB_OUTPUT"
  else
    echo "status=$1"
  fi
  exit 0
}

url="$(gcloud run services describe "$SERVICE" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --format='value(status.url)' 2>"$work/describe.err")" || url=""
if [ -z "$url" ]; then
  echo "::warning::Could not read the URL of '$SERVICE', so the new revision was not warmed: $(head -c 500 "$work/describe.err")"
  finish unavailable
fi

access_token="$(gcloud auth print-access-token 2>"$work/token.err")" || access_token=""
if [ -z "$access_token" ]; then
  echo "::warning::gcloud printed no access token, so the new revision was not warmed: $(head -c 500 "$work/token.err")"
  finish unavailable
fi

mint_status="$(curl -sS -o "$work/mint.json" -w '%{http_code}' --max-time 30 \
  -X POST "$iam_base/v1/projects/-/serviceAccounts/$DEPLOYER:generateIdToken" \
  -H "Authorization: Bearer $access_token" \
  -H "Content-Type: application/json" \
  --data "{\"audience\": \"$url\", \"includeEmail\": true}" 2>"$work/mint.err")" || mint_status="000"
case "$mint_status" in
  200) ;;
  401 | 403)
    echo "::warning::generateIdToken answered $mint_status for $DEPLOYER, so the new revision was not warmed. The deployer needs roles/iam.serviceAccountOpenIdTokenCreator on itself (infra apis stack)."
    finish forbidden
    ;;
  *)
    echo "::warning::generateIdToken answered $mint_status for $DEPLOYER, so the new revision was not warmed: $(cat "$work/mint.json" "$work/mint.err" 2>/dev/null | head -c 500)"
    finish unavailable
    ;;
esac
id_token="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["token"])' "$work/mint.json")"
echo "::add-mask::$id_token"

end=$((SECONDS + deadline_seconds))
covered=0
outcome=warm

for path in "${paths[@]}"; do
  covered=$((covered + 1))
  attempts=0
  while :; do
    attempts=$((attempts + 1))
    remaining=$((end - SECONDS))
    if [ "$remaining" -lt 1 ]; then
      remaining=1
    fi
    max_time=$((remaining < 60 ? remaining : 60))
    status="$(curl -sS -o "$work/body" -D "$work/headers" -w '%{http_code}' \
      --max-time "$max_time" \
      -H "X-Serverless-Authorization: Bearer $id_token" \
      "$url$path" 2>"$work/curl.err")" || status="000"

    case "$status" in
      200)
        echo "$path answered 200 after $attempts attempt(s)."
        break
        ;;
      401 | 403)
        echo "::warning::$path answered $status, so the new revision was not warmed. The deployer needs roles/run.invoker on '$SERVICE' (infra apis stack)."
        # The remaining paths would answer the same, so probing ends here,
        # and an error an earlier path answered still fails the deploy.
        if [ "$outcome" = error ]; then
          finish error
        fi
        finish forbidden
        ;;
      000 | 429 | 502 | 503 | 504)
        if [ "$SECONDS" -ge "$end" ]; then
          echo "::warning::$path still answered $status after $attempts attempt(s) and ${deadline_seconds} s; its build runs on for the next request."
          if [ "$outcome" = warm ]; then
            outcome=slow
          fi
          break
        fi
        retry_after="$(tr -d '\r' <"$work/headers" | awk -F': *' 'tolower($1) == "retry-after" && $2 ~ /^[0-9]+$/ { print $2; exit }')"
        wait_seconds="${retry_after:-5}"
        left=$((end - SECONDS))
        if [ "$wait_seconds" -gt "$left" ]; then
          wait_seconds="$left"
        fi
        if [ "$wait_seconds" -gt 0 ]; then
          sleep "$wait_seconds"
        fi
        ;;
      *)
        echo "::error::$path answered $status on the new revision of '$SERVICE': $(head -c 500 "$work/body")"
        outcome=error
        break
        ;;
    esac
  done
done

if [ "$covered" -ne "${#paths[@]}" ] || [ "$covered" -eq 0 ]; then
  echo "::error::warmed $covered of ${#paths[@]} paths."
  finish error
fi

finish "$outcome"
