#!/usr/bin/env bash
# Calls the apis service's warm paths on the new revision before traffic moves
# to it, so a revision that answers an error never serves, and the builds that
# run beside serving finish before readers arrive. The Search and
# Skills surfaces build the skills index, and the Graph surface builds the
# schema document and the UN terms body, after the lifespan returns. Cloud Run
# throttles the CPU of an instance with no request in flight, so a revision
# nobody calls stalls those builds until the first readers arrive and pay for
# them. Each warm path's handler waits on one of those builds:
#
#   /api/skills                        the skills index's first build
#                                      (`skills_api.deps.built_index`, up to
#                                      30 s, then 503)
#   /api/graph/schema                  the schema document (up to 10 s, then
#                                      503 `schema_building`)
#   /api/graph/schema/terms/un_system  the UN terms body (up to 10 s, then
#                                      503 `schema_building`)
#   POST /api/search/stated-scope      the stated-scope door, which a client
#                                      release calls; a probe that answers 200
#                                      shows the route is live
#
# A request keeps its instance's CPU allocated, so the builds run while the
# script waits, and a 503 the script retries means the build runs on.
#
# With PROBE_TAG set, the script probes the revision that Cloud Run tag names, at
# its tagged URL, which the deploy step created with no traffic. The tagged URL
# takes the service URL as its token audience. With the tag unset, or with no
# traffic entry carrying it, the script probes the service URL.
#
# The UN terms route answers 404 with `cause.code` `origin_not_served` when the
# deployment turns the UN origin off (`ORIGIN_CATALOGUE_UN_SYSTEM_ENABLED`).
# That answer means the route is not served here, so the script logs a skip
# and goes on. Any other 404 is an error.
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
#   PROBE_TAG    Cloud Run revision tag to probe. Optional.
#   REQUIRE_PROBE
#                `true` in production: a token mint or service answer of 401 or
#                403, or an unreadable URL or token, ends as `error` instead of
#                `forbidden` or `unavailable`, so the deploy stops before
#                traffic moves. Optional, defaults to `false`, where those
#                outcomes warn and the deploy goes on.
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
#                does until the infra grants apply; a warning, or `error`
#                under REQUIRE_PROBE. Probing stops
#                at the first 401 or 403, and the status is `error` instead
#                when an earlier path answered an error.
#   unavailable  the service URL or the token could not be read for another
#                reason; a warning, or `error` under REQUIRE_PROBE
#   error        the service answered a path with any other status; an
#                error annotation per path
#
# Exits 0 for every outcome. deploy.yml's next step fails the job on `error`. Exits non-zero only on a
# missing required variable.

set -euo pipefail

: "${SERVICE:?SERVICE is required}"
: "${REGION:?REGION is required}"
: "${PROJECT_ID:?PROJECT_ID is required}"
: "${DEPLOYER:?DEPLOYER is required}"
deadline_seconds="${WARM_DEADLINE_SECONDS:-600}"
iam_base="${IAM_CREDENTIALS_BASE:-https://iamcredentials.googleapis.com}"

probes=(
  "GET /api/skills"
  "GET /api/graph/schema"
  "GET /api/graph/schema/terms/un_system"
  "POST /api/search/stated-scope"
)
stated_scope_body='{"texts": ["total aid to Kenya in 2022"]}'
probe_tag="${PROBE_TAG:-}"
require_probe="${REQUIRE_PROBE:-false}"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

finish() {
  local status="$1"
  if [ "$require_probe" = true ] && { [ "$status" = forbidden ] || [ "$status" = unavailable ]; }; then
    echo "::error::The new revision could not be probed ($status), and this environment requires the probe before traffic moves."
    status=error
  fi
  echo "warm status=$status"
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "status=$status" >>"$GITHUB_OUTPUT"
  else
    echo "status=$status"
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

probe_url="$url"
if [ -n "$probe_tag" ]; then
  tagged="$(gcloud run services describe "$SERVICE" \
    --region="$REGION" \
    --project="$PROJECT_ID" \
    --format=json 2>"$work/describe-tag.err" |
    python3 -c '
import json, sys
tag = sys.argv[1]
for entry in json.load(sys.stdin).get("status", {}).get("traffic", []):
    if entry.get("tag") == tag and entry.get("url"):
        print(entry["url"])
        break
' "$probe_tag")" || tagged=""
  if [ -n "$tagged" ]; then
    probe_url="$tagged"
  else
    echo "::warning::No traffic entry of '$SERVICE' carries the tag '$probe_tag', so the service URL is probed: $(head -c 500 "$work/describe-tag.err")"
  fi
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

for probe in "${probes[@]}"; do
  method="${probe%% *}"
  path="${probe#* }"
  covered=$((covered + 1))
  curl_args=(-X "$method")
  if [ "$method" = POST ]; then
    curl_args+=(-H "Content-Type: application/json" --data "$stated_scope_body")
  fi
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
      "${curl_args[@]}" \
      -H "X-Serverless-Authorization: Bearer $id_token" \
      "$probe_url$path" 2>"$work/curl.err")" || status="000"

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
      404)
        code="$(python3 -c 'import json, sys
try:
    print(json.load(open(sys.argv[1])).get("cause", {}).get("code", ""))
except Exception:
    print("")' "$work/body")"
        if [ "$code" = origin_not_served ]; then
          echo "$path answered 404 origin_not_served: this deployment does not serve the route, so it is skipped."
          break
        fi
        echo "::error::$path answered 404 on the new revision of '$SERVICE': $(head -c 500 "$work/body")"
        outcome=error
        break
        ;;
      *)
        echo "::error::$path answered $status on the new revision of '$SERVICE': $(head -c 500 "$work/body")"
        outcome=error
        break
        ;;
    esac
  done
done

if [ "$covered" -ne "${#probes[@]}" ] || [ "$covered" -eq 0 ]; then
  echo "::error::warmed $covered of ${#probes[@]} paths."
  finish error
fi

finish "$outcome"
