# The services of the AssetOpsBench rows for the cluster jobs, sourced by functions.sh of CausalDiagnosis/cluster_jobs once
# env.sh has set its values: CouchDB, which the tool servers of AssetOpsBench read with --store couchdb, started by
# start_couchdb of start_couchdb.sh beside this file. A value that env.sh sets wins over the default here.
IMAGES=${IMAGES:-$ROOT/images}                        # the CouchDB image, pulled once
source "$(dirname "${BASH_SOURCE[0]}")/start_couchdb.sh"
# the scenario file of the anchored preset from the dataset ibm-research/AssetOpsBench, optional: no step of the setup reads
# it, since the digest job covers that preset without it and no grid file of the tables names it, and a run of the preset
# stops at its start with this command when the file is missing
OPTIONAL_DATA_FETCHES+=("ibm-research/AssetOpsBench python -m CausalDiagnosis.frameworks.AssetOpsBench.scenarios.download_scenarios")
IMAGE_PULLS+=(couchdb_image)                          # the CouchDB image, pulled by the image job of the setup
PLATFORM_RECORDS+=(couchdb_image)                     # the CouchDB image and its SHA-256 in the platform record
