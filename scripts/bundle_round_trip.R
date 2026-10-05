# Round-trips a generated Data Package v2 bundle through the R
# `frictionless` package (CRAN), as a companion to the Python round-trip
# `tests/materialise/test_bundle.py` runs in CI.
#
# Invoked by `.github/workflows/bundle-round-trip.yml` against the bundle
# directory `scripts/generate_sample_bundle.py` produces. Also runnable by
# hand against any bundle directory produced by
# `materialise.bundle.assemble_bundle`, as long as the descriptor's own
# resource paths point at real data files written beside `datapackage.json`.
#
# The pinned CRAN `frictionless` package (1.3.0) is v1-only: its `schema()`
# read path hard-fails unless each resource carries the legacy `profile:
# "tabular-data-resource"` property, so this script asserts that property is
# present alongside the v2 `type: "table"` (see `materialise.bundle`'s
# `DATAPACKAGE_PROFILE_URL` docstring for the full dual-publish rationale).
#
# Usage:
#   Rscript scripts/bundle_round_trip.R <path-to-bundle-directory>
#
# Requires the `frictionless` R package (`install.packages("frictionless")`).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("usage: Rscript scripts/bundle_round_trip.R <path-to-bundle-directory>")
}
bundle_dir <- args[[1]]

library(frictionless)

package <- read_package(file.path(bundle_dir, "datapackage.json"))

stopifnot(identical(package$`$schema`, "https://datapackage.org/profiles/2.0/datapackage.json"))
stopifnot(length(package$resources) >= 1)

for (resource in package$resources) {
  stopifnot(identical(resource$type, "table"))
  stopifnot(identical(resource$profile, "tabular-data-resource"))
  stopifnot(!is.null(resource$hash))
  stopifnot(!is.null(resource$bytes))
  stopifnot(!is.null(resource$schema))

  data <- read_resource(package, resource$name)
  cat(sprintf(
    "resource %s: %d rows, %d columns, hash %s\n",
    resource$name, nrow(data), ncol(data), resource$hash
  ))
}

cat("bundle_round_trip.R: OK\n")
