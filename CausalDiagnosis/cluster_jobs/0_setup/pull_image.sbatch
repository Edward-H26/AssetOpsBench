#!/bin/bash
#SBATCH -J cd-image
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
# Pulls the SGLang serving image into $SIF of env.sh on a CPU node, the way the project's copy was pulled on 2026-10-08:
# docker://lmsysorg/sglang:v0.5.19-cu129, the newest SGLang release with a CUDA 12 image at that date (the later releases
# shipped CUDA 13 images, which the node drivers of that date could not run). The OCI layers and the unpacked root file
# system go to node local disk, and only the finished SIF lands on scratch. Rerunning skips
# the pull, and with the shared image of env.sh readable the job ends at once. The tag may move on the Hub; the serving job writes the image's SHA-256 and the SGLang version into every
# endpoint file, so a later pull is visible in the run ledgers. Submit as
#   bash $H/submit.sh pull_image.sbatch
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"
IMAGE=${IMAGE:-lmsysorg/sglang:v0.5.19-cu129}
PULL_FREE_GB=${PULL_FREE_GB:-100}                     # the free space of node local disk that the pull needs, in GB
if [ -n "$SHARED" ] && [[ "$SIF" == "$SHARED"/* ]]; then echo "the shared image $SIF is readable; nothing to pull, leave SHARED empty in env.sh for a copy of your own"; echo "CD_IMAGE_DONE $SIF"; exit 0; fi
module load apptainer/1.4.5
L=$(node_local_dir); trap 'rm -rf "$L"' EXIT
free_gb=$(df -BG --output=avail "$L" | tail -1 | tr -dc 0-9)
[ "${free_gb:-0}" -ge "$PULL_FREE_GB" ] || { echo "only ${free_gb:-0} GB free on node local disk at $L, the pull needs $PULL_FREE_GB GB" >&2; exit 2; }
export APPTAINER_CACHEDIR=$L/apptainer-cache APPTAINER_TMPDIR=$L/apptainer
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$SIF")"
job_header "image $IMAGE into $SIF"; df -h "$(dirname "$SIF")" "$L" | cat
if [ ! -s "$SIF" ]; then
  apptainer pull "$SIF.partial" "docker://$IMAGE" && mv "$SIF.partial" "$SIF" || { rm -f "$SIF.partial"; exit 1; }
fi
ls -la "$SIF"
# the tag v<version>[-<suffix>] names the SGLang version that the image must hold, so a tag that names none stops the job
WANT=${IMAGE##*:}; WANT=${WANT#v}; WANT=${WANT%%-*}
VERSION=$(apptainer exec "$SIF" python3 -c "import sglang; print(sglang.__version__)") || { echo "the image $SIF cannot import sglang" >&2; exit 3; }
echo "sglang $VERSION"
[ "$VERSION" = "$WANT" ] || { echo "the image $SIF holds SGLang $VERSION, not the version $WANT that its tag $IMAGE names" >&2; exit 3; }
echo "CD_IMAGE_DONE $SIF"
