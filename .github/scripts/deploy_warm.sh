#!/usr/bin/env bash
# Probes one named apis revision at its tagged URL before traffic moves to it,
# so a revision that answers an error never serves, and the builds that run
# beside serving finish before readers arrive. The Search and Skills surfaces
# build the skills index, and the Graph surface builds the schema document and
# the UN terms body, after the lifespan returns. Cloud Run throttles the CPU of
# an instance with no request in flight, so a revision nobody calls stalls
# those builds until the first readers arrive and pay for them. The probes:
#
#   /api/skills                        the skills index's first build
#                                      (`skills_api.deps.built_index`, up to
#                                      30 s, then 503)
#   /api/graph/schema                  the schema document (up to 10 s, then
#                                      503 `schema_building`)
#   /api/graph/schema/terms/un_system  the UN terms body (up to 10 s, then
#                                      503 `schema_building`)
#   POST /api/search/stated-scope      the stated-scope route, which runs
#                                      regex reads only
#
# A request keeps its instance's CPU allocated, so the builds run while the
# script waits, and a 503 the script retries means the build runs on. Each
# route has its own deadline, so a slow route leaves the next one its full
# budget.
#
# The revision is reached at the URL of the traffic entry that carries
# PROBE_TAG and names REVISION, the revision this run staged, which
# is the one the traffic step moves traffic to. Without that entry, for example
# when the tag sits on another revision, the script ends as `error` and probes
# nothing, since any other URL reaches a different revision. The
# token's audience is the service URL, which Cloud Run requires for a tagged
# URL too. The script decodes the minted token's `aud` claim, ends as `error`
# when it is not the service URL that describe returned, and logs the match.
# A 401 or 403 from a route after that match means Cloud Run refused the token
# on the tagged URL or the deployer lacks roles/run.invoker.
#
# The UN terms route answers 404 with `cause.code` `origin_not_served` when the
# deployment turns the UN origin off (`ORIGIN_CATALOGUE_UN_SYSTEM_ENABLED`).
# That answer from that route means the route is not served here, so the
# script logs a skip and goes on. Any other 404 is an error.
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
#   REVISION     the revision to probe -- required
#   PROBE_TAG    the Cloud Run traffic tag on REVISION -- required
#   REQUIRE_PROBE
#                `true` in production: every route must answer 200 within its
#                deadline, so `slow`, `forbidden` and `unavailable` end as
#                `error` and the deploy stops before traffic moves. Optional,
#                defaults to `false`, where those outcomes warn and the
#                deploy goes on.
#   ROUTE_DEADLINE_SECONDS
#                how long each route may take, retries included. Optional,
#                defaults to 300.
#   IAM_CREDENTIALS_BASE
#                IAM Credentials API root. Optional; tests point it at a
#                local server.
#
# Writes `status=<outcome>` to $GITHUB_OUTPUT (or stdout, when $GITHUB_OUTPUT
# is unset, so this runs the same way under a test):
#
#   warm         every route answered 200
#   slow         a route still answered 502, 503, 504 or 429 at its
#                deadline. A warning, since the builds run on for the next
#                request, or `error` under REQUIRE_PROBE
#   forbidden    the token mint or the revision answered 401 or 403, as it
#                does until the infra grants apply. A warning, or `error`
#                under REQUIRE_PROBE. Probing stops at the first 401 or 403,
#                and the status is `error` when an earlier route answered an
#                error.
#   unavailable  the service URL or the access token could not be read, or
#                the token mint answered another status than 200, 401 or
#                403, after up to 4 attempts for a 429, a 5xx or no answer.
#                A warning, or `error`
#                under REQUIRE_PROBE
#   error        the revision answered a route with any other status, a
#                route never answered at all within its deadline (probing
#                stops there), no traffic entry carries PROBE_TAG on
#                REVISION, or the token's audience is not the service URL
#
# Exits 0 for every outcome, and deploy.yml's next step fails the job on
# `error`.
# Exits non-zero only on a missing required variable.

set -euo pipefail

: "${SERVICE:?SERVICE is required}"
: "${REGION:?REGION is required}"
: "${PROJECT_ID:?PROJECT_ID is required}"
: "${DEPLOYER:?DEPLOYER is required}"
: "${REVISION:?REVISION is required}"
: "${PROBE_TAG:?PROBE_TAG is required}"
route_seconds="${ROUTE_DEADLINE_SECONDS:-300}"
iam_base="${IAM_CREDENTIALS_BASE:-https://iamcredentials.googleapis.com}"
require_probe="${REQUIRE_PROBE:-false}"

probes=(
  "GET /api/skills"
  "GET /api/graph/schema"
  "GET /api/graph/schema/terms/un_system"
  "POST /api/search/stated-scope"
)
un_terms=/api/graph/schema/terms/un_system
stated_scope_body='{"texts": ["total aid to Kenya in 2022"]}'

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

finish() {
  local status="$1"
  if [ "$require_probe" = true ] && [ "$status" != warm ] && [ "$status" != error ]; then
    echo "::error::Revision $REVISION ended the probe as $status, and this environment requires every route to answer 200 before traffic moves."
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

# Waits out a retry: the Retry-After seconds in the given headers file, or 5,
# and no later than the given deadline.
pause() {
  local headers="$1" until="$2" wait_seconds retry_after
  retry_after="$(tr -d '\r' <"$headers" 2>/dev/null | awk -F': *' 'tolower($1) == "retry-after" && $2 ~ /^[0-9]+$/ { print $2; exit }')"
  wait_seconds="${retry_after:-5}"
  if [ "$wait_seconds" -gt $((until - SECONDS)) ]; then
    wait_seconds=$((until - SECONDS))
  fi
  if [ "$wait_seconds" -gt 0 ]; then
    sleep "$wait_seconds"
  fi
}

described="$(gcloud run services describe "$SERVICE" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --format=json 2>"$work/describe.err")" || described=""
# Two lines: the service URL, and the URL of the traffic entry carrying the tag
# on the revision. A missing one prints as an empty line.
urls="$(python3 -c '
import json, sys
try:
    status = json.loads(sys.argv[1]).get("status") or {}
except ValueError:
    status = {}
tagged = [
    e.get("url")
    for e in status.get("traffic") or []
    if e.get("tag") == sys.argv[2] and e.get("revisionName") == sys.argv[3] and e.get("url")
]
print(status.get("url") or "")
print(tagged[0] if tagged else "")
' "$described" "$PROBE_TAG" "$REVISION")"
url="$(sed -n 1p <<<"$urls")"
probe_url="$(sed -n 2p <<<"$urls")"
if [ -z "$url" ]; then
  echo "::warning::Could not read the URL of '$SERVICE', so revision $REVISION was not probed: $(head -c 500 "$work/describe.err")"
  finish unavailable
fi
if [ -z "$probe_url" ]; then
  echo "::error::No traffic entry of '$SERVICE' carries the tag '$PROBE_TAG' on revision $REVISION, so it cannot be probed, and any other URL reaches a different revision."
  finish error
fi
echo "Probing revision $REVISION at the URL of its '$PROBE_TAG' tag."

access_token="$(gcloud auth print-access-token 2>"$work/token.err")" || access_token=""
if [ -z "$access_token" ]; then
  echo "::warning::gcloud printed no access token, so revision $REVISION was not probed: $(head -c 500 "$work/token.err")"
  finish unavailable
fi

# generateIdToken retries an answer of 429 or 5xx, or none, up to 4 attempts.
mint_attempts=0
while :; do
  mint_attempts=$((mint_attempts + 1))
  mint_status="$(curl -sS -o "$work/mint.json" -D "$work/mint.headers" -w '%{http_code}' --max-time 30 \
    -X POST "$iam_base/v1/projects/-/serviceAccounts/$DEPLOYER:generateIdToken" \
    -H "Authorization: Bearer $access_token" \
    -H "Content-Type: application/json" \
    --data "{\"audience\": \"$url\", \"includeEmail\": true}" 2>"$work/mint.err")" || mint_status="000"
  echo "generateIdToken attempt $mint_attempts answered $mint_status."
  case "$mint_status" in
    200) break ;;
    401 | 403)
      echo "::warning::generateIdToken answered $mint_status for $DEPLOYER, so revision $REVISION was not probed. The deployer needs roles/iam.serviceAccountOpenIdTokenCreator on itself (infra apis stack)."
      finish forbidden
      ;;
    000 | 429 | 5??)
      if [ "$mint_attempts" -lt 4 ]; then
        pause "$work/mint.headers" $((SECONDS + 30))
        continue
      fi
      ;;
  esac
  echo "::warning::generateIdToken answered $mint_status for $DEPLOYER after $mint_attempts attempt(s), so revision $REVISION was not probed: $(cat "$work/mint.json" "$work/mint.err" 2>/dev/null | head -c 500)"
  finish unavailable
done

# The token, then `yes` or `no` for whether its `aud` claim is the service URL.
minted="$(python3 -c '
import base64, json, sys
token = json.load(open(sys.argv[1]))["token"]
print(token)
try:
    payload = token.split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    print("yes" if claims.get("aud") == sys.argv[2] else "no")
except (IndexError, ValueError):
    print("no")
' "$work/mint.json" "$url")"
id_token="$(sed -n 1p <<<"$minted")"
echo "::add-mask::$id_token"
if [ "$(sed -n 2p <<<"$minted")" = yes ]; then
  echo "ID token aud claim: the service URL that describe returned."
else
  echo "::error::The ID token generateIdToken returned has an aud claim other than the service URL that describe returned, which Cloud Run requires on a tagged URL."
  finish error
fi

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
  end=$((SECONDS + route_seconds))
  attempts=0
  answered=false
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
    echo "$method $path attempt $attempts answered $status."
    if [ "$status" != 000 ]; then
      answered=true
    fi

    case "$status" in
      200)
        break
        ;;
      401 | 403)
        echo "::warning::$path answered $status to a token whose aud claim is the service URL, so Cloud Run refused the token on the tagged URL, or the deployer lacks roles/run.invoker on '$SERVICE' (infra apis stack). Revision $REVISION was not probed."
        # The remaining routes would answer the same, so probing ends here,
        # and an error an earlier route answered still fails the deploy.
        if [ "$outcome" = error ]; then
          finish error
        fi
        finish forbidden
        ;;
      000 | 429 | 502 | 503 | 504)
        if [ "$SECONDS" -ge "$end" ] && [ "$answered" = false ]; then
          echo "::error::$path never answered in $attempts attempt(s) and ${route_seconds} s, so revision $REVISION hangs; probing stops here."
          finish error
        fi
        if [ "$SECONDS" -ge "$end" ]; then
          echo "::warning::$path still answered $status after $attempts attempt(s) and ${route_seconds} s; its build runs on for the next request."
          if [ "$outcome" = warm ]; then
            outcome=slow
          fi
          break
        fi
        pause "$work/headers" "$end"
        ;;
      404)
        code="$(python3 -c 'import json, sys
try:
    print(json.load(open(sys.argv[1])).get("cause", {}).get("code", ""))
except Exception:
    print("")' "$work/body")"
        if [ "$path" = "$un_terms" ] && [ "$code" = origin_not_served ]; then
          echo "$path answered 404 origin_not_served: this deployment does not serve the route, so it is skipped."
          break
        fi
        echo "::error::$path answered 404 on revision $REVISION: $(head -c 500 "$work/body")"
        outcome=error
        break
        ;;
      *)
        echo "::error::$path answered $status on revision $REVISION: $(head -c 500 "$work/body")"
        outcome=error
        break
        ;;
    esac
  done
done

if [ "$covered" -ne "${#probes[@]}" ] || [ "$covered" -eq 0 ]; then
  echo "::error::probed $covered of ${#probes[@]} routes."
  finish error
fi

finish "$outcome"
