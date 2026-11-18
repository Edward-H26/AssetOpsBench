#!/bin/bash
#SBATCH -J cd-image
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
# Pulls the SGLang serving image into $SIF of env.sh on a CPU node, the way the project's copy was pulled on 2026-10-08:
# docker://lmsysorg/sglang:v0.5.19-cu129, the newest SGLang release with a CUDA 12 image at that date (the later releases
# shipped CUDA 13 images, which the node drivers of that date could not run). The pull is pull_image_once of functions.sh:
# the root file system is unpacked on node local disk, the layers wait in a cache on scratch until the image is in place, so
# that a pull that a broken link or the time limit ended goes on from the layers it finished, and only the finished SIF stays
# on scratch. Rerunning skips the pull, and with the shared image of env.sh readable the job pulls no SGLang image. The tag
# may move on the Hub; the serving job writes the image's SHA-256 and the SGLang version into every endpoint file, so a later
# pull is visible in the run ledgers. The job then calls every function of IMAGE_PULLS, the images that the frameworks of the
# tree start, such as the CouchDB image of the AssetOpsBench rows, so that no later job pulls an image while it holds a card.
# Submit as
#   bash $H/submit.sh pull_image.sbatch
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"
IMAGE=${IMAGE:-lmsysorg/sglang:v0.5.19-cu129}
PULL_FREE_GB=${PULL_FREE_GB:-100}                     # the free space of node local disk that the pull needs, in GB
L=$(node_local_dir); trap 'rm -rf "$L"' EXIT
# loaded before the export below, since a modulefile may set APPTAINER_CACHEDIR and APPTAINER_TMPDIR itself on its load
module load apptainer/1.4.5
# every apptainer command of the job unpacks and caches on node local disk, the version check of an image already in place
# included: without squashfuse apptainer unpacks the whole image for an exec, which the shared /tmp of a node cannot hold
export APPTAINER_CACHEDIR=$L/apptainer-cache APPTAINER_TMPDIR=$L/apptainer
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
job_header "image $IMAGE into $SIF"

sglang_image() {
  # the SGLang image of the serving jobs, pulled once, and the check that it holds the SGLang version its tag names
  if [ -n "$SHARED" ] && [[ "$SIF" == "$SHARED"/* ]]; then echo "the shared image $SIF is readable; nothing to pull, leave SHARED empty in env.sh for a copy of your own"; return 0; fi
  local free_gb want version
  free_gb=$(df -BG --output=avail "$L" | tail -1 | tr -dc 0-9)
  [ "${free_gb:-0}" -ge "$PULL_FREE_GB" ] || { echo "only ${free_gb:-0} GB free on node local disk at $L, the pull needs $PULL_FREE_GB GB" >&2; exit 2; }
  mkdir -p "$(dirname "$SIF")"
  df -h "$(dirname "$SIF")" "$L" | cat
  pull_image_once "$SIF" "docker://$IMAGE" "$L" || exit 1
  ls -la "$SIF"
  # the tag v<version>[-<suffix>] names the SGLang version that the image must hold, so a tag that names none stops the job
  want=${IMAGE##*:}; want=${want#v}; want=${want%%-*}
  version=$(apptainer exec "$SIF" python3 -c "import sglang; print(sglang.__version__)") || { echo "the image $SIF cannot import sglang" >&2; exit 3; }
  echo "sglang $version"
  [ "$version" = "$want" ] || { echo "the image $SIF holds SGLang $version, not the version $want that its tag $IMAGE names" >&2; exit 3; }
}

sglang_image
for pull in ${IMAGE_PULLS[@]+"${IMAGE_PULLS[@]}"}; do "$pull" "$L" || exit 3; done
echo "CD_IMAGE_DONE $SIF"
