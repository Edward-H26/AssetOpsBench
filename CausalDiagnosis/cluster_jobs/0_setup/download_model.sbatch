#!/bin/bash
#SBATCH -J cd-download
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
# Downloads the model MODEL from Hugging Face into $MODELS/<name> on a CPU node, since compute nodes reach
# huggingface.co and the login node should not carry the transfer. The Hub cache lives on node local disk so that the
# weights exist once, in $MODELS on scratch, since the home filesystem has a small quota. The job fetches the Hub commit that
# the tracked serving file of record pins for MODEL (model_revision of env.sh), or HF_REVISION given as a full commit, refuses a model without one, takes a shared
# copy only when it holds that commit, and stops when the files it fetched come from another commit. Check the free space
# before submitting (df -h $MODELS): a model in BF16 needs two bytes per parameter. Submit as
#   bash $H/submit.sh download_model.sbatch                                  (the default MODEL of env.sh)
#   MODEL=<Hugging Face id> bash $H/submit.sh download_model.sbatch          (another model, for example the judge)
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"
require_model
L=$(node_local_dir); trap 'rm -rf "$L"' EXIT
export HF_HOME=$L/hf UV_CACHE_DIR=$L/uv-cache TMPDIR=$L/tmp; mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
export PATH=$(dirname "$UV_BIN"):$PATH
TARGET=$MODELS/${MODEL##*/}
# the check of the hashed files against the Hub's listing: every large file of the listing must be in the folder with its SHA-256
LISTING_CHECK='import json, sys
listing = {e["path"]: e["lfs"]["oid"] for e in json.load(open(sys.argv[1])) if e.get("type") == "file" and e.get("lfs")}
found = dict(reversed(line.rstrip("\n").split("  ", 1)) for line in open(sys.argv[2]) if line.strip())
bad = sorted(path for path, oid in listing.items() if found.get(path) != oid)
print(f"FILES_CHECKED {len(listing)} large files against the listing of the Hub")
sys.exit(f"the large files {bad[:5]} differ from the listing of the Hub or are missing" if bad else 0)'
if [[ "$MODEL" == */* && -d "$MODEL" ]]; then echo "MODEL is a model folder already on the cluster: $MODEL; nothing to download"; echo "CD_DOWNLOAD_DONE $MODEL"; exit 0; fi
HF_REVISION=${HF_REVISION:-$(model_revision "$MODEL")}
[[ "$HF_REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "no pinned commit for $MODEL, so nothing is downloaded: pin the sha of https://huggingface.co/api/models/$MODEL in the serving file of record, CausalDiagnosis/experiments/table_definitions/serving_of_record.json" >&2; exit 2; }
if [ "$(head -n 1 "$TARGET/HUB_COMMIT.txt" 2>/dev/null)" = "$HF_REVISION" ] && [ -s "$TARGET/FILES.sha256" ]; then
  echo "$TARGET holds the pinned commit $HF_REVISION with its hashed files; nothing to download"; echo "CD_DOWNLOAD_DONE $TARGET"; exit 0
fi
SHARED_COPY=$SHARED/models/${MODEL##*/}
if [ ! -f "$TARGET/config.json" ] && [ -n "$SHARED" ] && [ -r "$SHARED_COPY/config.json" ]; then
  COMMIT=$(head -n 1 "$SHARED_COPY/HUB_COMMIT.txt" 2>/dev/null)
  if [ -z "$COMMIT" ]; then
    # a copy that records no commit says nothing of its files, so it is not used, and the job fetches a copy of its own at the pin
    echo "the shared copy $SHARED_COPY records no commit, so the job downloads its own copy at the pinned $HF_REVISION"
  else
    [ "$COMMIT" = "$HF_REVISION" ] || { echo "the shared copy $SHARED_COPY holds the commit $COMMIT, not the pinned $HF_REVISION: pin its commit in the serving file of record, or leave SHARED empty in env.sh for a copy of your own" >&2; exit 3; }
    echo "the shared copy $SHARED_COPY holds the pinned commit $HF_REVISION; nothing to download"; echo "CD_DOWNLOAD_DONE $SHARED_COPY"; exit 0
  fi
fi
job_header "model $MODEL into $TARGET"; df -h "$MODELS" | cat
[ -n "${HF_TOKEN:-}" ] && echo "HF_TOKEN set (gated model)"
# hf is the command line tool of huggingface_hub; uvx runs it in an ephemeral environment on node local disk
uvx --from huggingface_hub hf download "$MODEL" --local-dir "$TARGET" --revision "$HF_REVISION" \
  || { echo "the download of $MODEL into $TARGET failed with status $?, so no HUB_COMMIT.txt is written" >&2; exit 3; }
ls -la "$TARGET" | head -40
du -sh "$TARGET" | cat
# the Hub commit of the files, for the run header: the first line of the metadata of every file, which must be the pin alone
COMMITS=$(for f in "$TARGET"/.cache/huggingface/download/*.metadata; do head -n 1 "$f" 2>/dev/null; done | sort -u)
[ "$COMMITS" = "$HF_REVISION" ] || { echo "the files of $TARGET come from the commits ${COMMITS//$'\n'/ }, not the pinned $HF_REVISION alone" >&2; exit 3; }
# FILES.sha256: one sha256sum line per file of the folder, which the endpoint file of every serving job hashes into the platform
# record; each file the Hub keeps as a large file must have the SHA-256 that the Hub's listing of the pin names for it
( cd "$TARGET" && find . -type f ! -path './.cache/*' ! -name HUB_COMMIT.txt ! -name 'FILES.sha256*' | sed 's|^\./||' | LC_ALL=C sort \
  | while IFS= read -r f; do sha256sum "$f"; done ) > "$TARGET/FILES.sha256.partial" || { echo "the files of $TARGET could not be hashed" >&2; exit 3; }
curl -sf ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} "https://huggingface.co/api/models/$MODEL/tree/$HF_REVISION?recursive=true" > "$L/listing.json" \
  || { echo "the Hub's listing of $MODEL at $HF_REVISION could not be read, so the files are not checked" >&2; exit 3; }
python3 -c "$LISTING_CHECK" "$L/listing.json" "$TARGET/FILES.sha256.partial" || exit 3
mv "$TARGET/FILES.sha256.partial" "$TARGET/FILES.sha256"
echo "$HF_REVISION" > "$TARGET/HUB_COMMIT.txt"
echo "CD_DOWNLOAD_DONE $TARGET"
