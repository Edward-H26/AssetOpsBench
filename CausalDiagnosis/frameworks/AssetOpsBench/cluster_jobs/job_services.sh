# The services of the AssetOpsBench rows for the cluster jobs, sourced by functions.sh of CausalDiagnosis/cluster_jobs once
# env.sh has set its values: CouchDB, which the tool servers of AssetOpsBench read with --store couchdb, started by
# start_couchdb of start_couchdb.sh beside this file. A value that env.sh sets wins over the default here.
IMAGES=${IMAGES:-$ROOT/images}                        # the CouchDB image, pulled once
source "$(dirname "${BASH_SOURCE[0]}")/start_couchdb.sh"
DATA_FETCHES+=("python -m CausalDiagnosis.frameworks.AssetOpsBench.scenarios.download_scenarios")   # the scenario file of the anchored preset
IMAGE_PULLS+=(couchdb_image)                          # the CouchDB image, pulled by the image job of the setup
PLATFORM_RECORDS+=(couchdb_image)                     # the CouchDB image and its SHA-256 in the platform record
