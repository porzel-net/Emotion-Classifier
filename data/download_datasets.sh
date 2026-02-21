#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DATA_DIR="$SCRIPT_DIR"

FORCE_DOWNLOAD=0
FORCE_UNZIP=0
ONLY_DATASET=""

usage() {
  cat <<'USAGE'
Usage: sh data/download_datasets.sh [options]

Downloads and unzips project datasets into data/ with stable folder names.
Existing downloads/extractions are skipped by default.

Options:
  --dataset <name>      Download only one dataset.
                        Allowed: fer2013, landmarks68, soloface, affectnet
  --force-download      Re-download zip even if it already exists.
  --force-unzip         Re-extract even if target folder already looks complete.
  -h, --help            Show this help.
USAGE
}

require_cmd() {
  cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: Required command not found: $cmd" >&2
    exit 1
  fi
}

dataset_config() {
  dataset="$1"
  case "$dataset" in
    fer2013)
      echo "FER-2013|fer2013.zip|https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013|fer2013|fer2013|train,test|emotion-detection-fer"
      ;;
    landmarks68)
      echo "68-landmark keypoint dataset|cropped-face-keypoint-dataset-68-landmarks.zip|https://www.kaggle.com/api/v1/datasets/download/sovitrath/cropped-face-keypoint-dataset-68-landmarks|cropped-face-keypoint-dataset-68-landmarks|cropped-face-keypoint-dataset-68-landmarks|training.csv,test.csv|"
      ;;
    soloface)
      echo "SoloFace detection dataset|soloface-detection-dataset.zip|https://zenodo.org/records/14474899/files/soloface-detection-dataset.zip?download=1|soloface-detection-dataset|soloface-detection-dataset|train/images,test/images,val/images|"
      ;;
    affectnet)
      echo "AffectNet YOLO format|affectnet-yolo-format.zip|https://www.kaggle.com/api/v1/datasets/download/fatihkgg/affectnet-yolo-format|affectnet-yolo-format|affectnet-yolo-format||"
      ;;
    *)
      echo "ERROR: Unknown dataset '$dataset'." >&2
      exit 1
      ;;
  esac
}

is_nonempty_dir() {
  dir_path="$1"
  [ -d "$dir_path" ] || return 1
  if find "$dir_path" -mindepth 1 -print -quit | grep -q .; then
    return 0
  fi
  return 1
}

is_dataset_ready() {
  target_dir="$1"
  markers_csv="$2"

  [ -d "$target_dir" ] || return 1

  if [ -z "$markers_csv" ]; then
    is_nonempty_dir "$target_dir"
    return $?
  fi

  old_ifs="$IFS"
  IFS=','
  for marker in $markers_csv; do
    if [ ! -e "$target_dir/$marker" ]; then
      IFS="$old_ifs"
      return 1
    fi
  done
  IFS="$old_ifs"

  return 0
}

migrate_legacy_target_if_needed() {
  label="$1"
  target_path="$2"
  legacy_path="$3"
  markers_csv="$4"

  [ -n "$legacy_path" ] || return 0

  if is_dataset_ready "$target_path" "$markers_csv"; then
    return 0
  fi

  if ! is_dataset_ready "$legacy_path" "$markers_csv"; then
    return 0
  fi

  echo "[$label] found legacy dataset folder: $legacy_path"

  if [ ! -e "$target_path" ]; then
    mv "$legacy_path" "$target_path"
    echo "[$label] renamed legacy folder to: $target_path"
    return 0
  fi

  cp -a "$legacy_path"/. "$target_path"/
  echo "[$label] merged legacy folder into: $target_path"
}

download_zip_if_needed() {
  label="$1"
  url="$2"
  zip_path="$3"
  target_path="$4"
  markers_csv="$5"

  if [ "$FORCE_DOWNLOAD" -eq 0 ] && [ -s "$zip_path" ]; then
    echo "[$label] zip already present: $zip_path"
    return 0
  fi

  if [ "$FORCE_DOWNLOAD" -eq 0 ] && is_dataset_ready "$target_path" "$markers_csv"; then
    echo "[$label] already extracted at: $target_path (skip download)"
    return 0
  fi

  echo "[$label] downloading..."
  curl -fL --retry 3 --retry-delay 2 -o "$zip_path" "$url"
}

detect_extracted_root() {
  extract_dir="$1"
  source_hint="$2"
  target_hint="$3"

  if [ -n "$source_hint" ] && [ -d "$extract_dir/$source_hint" ]; then
    echo "$extract_dir/$source_hint"
    return 0
  fi

  if [ -n "$target_hint" ] && [ -d "$extract_dir/$target_hint" ]; then
    echo "$extract_dir/$target_hint"
    return 0
  fi

  first_dir=$(find "$extract_dir" -mindepth 1 -maxdepth 1 -type d | sed -n '1p')
  second_dir=$(find "$extract_dir" -mindepth 1 -maxdepth 1 -type d | sed -n '2p')

  if [ -n "$first_dir" ] && [ -z "$second_dir" ]; then
    echo "$first_dir"
    return 0
  fi

  # Fallback for flat zip structures.
  echo "$extract_dir"
}

unzip_if_needed() {
  label="$1"
  zip_path="$2"
  target_path="$3"
  source_hint="$4"
  markers_csv="$5"

  if [ "$FORCE_UNZIP" -eq 0 ] && is_dataset_ready "$target_path" "$markers_csv"; then
    echo "[$label] target already ready: $target_path"
    return 0
  fi

  if [ ! -s "$zip_path" ]; then
    echo "ERROR: [$label] Missing zip for extraction: $zip_path" >&2
    exit 1
  fi

  safe_label=$(echo "$label" | tr ' ' '_')
  extract_tmp=$(mktemp -d "$DATA_DIR/.extract.${safe_label}.XXXXXX")
  unzip -q -o "$zip_path" -d "$extract_tmp"

  source_path=$(detect_extracted_root "$extract_tmp" "$source_hint" "$(basename "$target_path")")
  mkdir -p "$target_path"
  cp -a "$source_path"/. "$target_path"/
  rm -rf "$extract_tmp"

  if is_dataset_ready "$target_path" "$markers_csv"; then
    echo "[$label] ready at: $target_path"
  else
    echo "ERROR: [$label] extraction finished, but expected files are missing in $target_path" >&2
    exit 1
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: Missing value for --dataset" >&2
        usage
        exit 1
      fi
      ONLY_DATASET="$2"
      shift 2
      ;;
    --force-download)
      FORCE_DOWNLOAD=1
      shift
      ;;
    --force-unzip)
      FORCE_UNZIP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument '$1'" >&2
      usage
      exit 1
      ;;
  esac
done

require_cmd curl
require_cmd unzip

DATASETS="fer2013 landmarks68 soloface affectnet"
if [ -n "$ONLY_DATASET" ]; then
  DATASETS="$ONLY_DATASET"
fi

echo "Data directory: $DATA_DIR"
for dataset in $DATASETS; do
  cfg=$(dataset_config "$dataset")

  label=$(echo "$cfg" | cut -d '|' -f1)
  zip_name=$(echo "$cfg" | cut -d '|' -f2)
  url=$(echo "$cfg" | cut -d '|' -f3)
  target_name=$(echo "$cfg" | cut -d '|' -f4)
  source_hint=$(echo "$cfg" | cut -d '|' -f5)
  markers_csv=$(echo "$cfg" | cut -d '|' -f6)
  legacy_name=$(echo "$cfg" | cut -d '|' -f7)

  zip_path="$DATA_DIR/$zip_name"
  target_path="$DATA_DIR/$target_name"

  legacy_path=""
  if [ -n "$legacy_name" ]; then
    legacy_path="$DATA_DIR/$legacy_name"
  fi

  migrate_legacy_target_if_needed "$label" "$target_path" "$legacy_path" "$markers_csv"
  download_zip_if_needed "$label" "$url" "$zip_path" "$target_path" "$markers_csv"
  unzip_if_needed "$label" "$zip_path" "$target_path" "$source_hint" "$markers_csv"
done

echo "All requested datasets are ready."
