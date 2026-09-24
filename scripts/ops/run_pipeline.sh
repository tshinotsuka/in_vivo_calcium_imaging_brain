#!/usr/bin/env bash
# run_pipeline.sh — the whole analysis, over any number of datasets.
#
# Takes a list, a glob, or a date, and runs the same stages on each in turn. A
# dataset that fails does not stop the others: the failure is recorded and
# reported in a table at the end, because the alternative is discovering at the
# end of an overnight run that everything after the first bad subject never
# ran.
#
# Stages are named and selectable because they differ by orders of magnitude in
# cost: detection takes minutes, the denoisers an hour each, and the figures
# seconds and get redrawn constantly. Each runs in the environment its tool
# needs, so none has to be active beforehand and none is left active after.
#
# Everything lands under <dataset>/work, the disposable layer. Nothing here
# writes to raw/, and results/ is touched only by the movie stage.
#
# Usage:
#   bash run_pipeline.sh -d <dataset> [-d <dataset> ...] [-s stages] [-n]
#   bash run_pipeline.sh --glob 'sub-sk5*'      [-s stages]
#   bash run_pipeline.sh --date 20260918        [-s stages]
#   bash run_pipeline.sh --all                  [-s stages]
#
#   stages: meta detect qc auc figs spikes movie cnmf deepcad foopsi compare
#   default: meta,detect,qc,auc,figs
#
# Examples:
#   bash run_pipeline.sh -d 20260917_sub-sk29_ses-01 -d 20260918_sub-sk53_ses-01
#   bash run_pipeline.sh --date 20260918 -s all -n
#   bash run_pipeline.sh --glob '*sk5*' -s figs --roi '4 11 17'

set -uo pipefail

IVWIB="${IVWIB_REPO:-/media/tshino/DATA/Projects/in_vivo_water_imaging_brain}"
WS="${WS:-/media/tshino/DATA/workspace/2026_aqp4ko_water_intoxication/2photon}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Ask git where the repository root is rather than counting directories up from
# the script: a dirname chain silently breaks the moment the script is moved,
# which is exactly how it broke.
REPO="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "$REPO" ] || REPO="$(cd "$HERE/../.." && pwd)"
SCRIPTS="$REPO/scripts"
[ -f "$SCRIPTS/run_event_auc.py" ] || {
  echo "ERROR: no scripts/ at $SCRIPTS (resolved from $HERE)" >&2; exit 2; }

ENV_META="${ENV_META:-ivwib}"
ENV_S2P="${ENV_S2P:-caimg}"
ENV_CAIMAN="${ENV_CAIMAN:-caiman}"
ENV_DEEPCAD="${ENV_DEEPCAD:-deepcadrt}"

declare -a WANTED=()
STAGES="meta,detect,qc,auc,figs"
DRYRUN=0
PATTERN=""
DEVICE="${TORCH_DEVICE:-cuda}"
DIAMETER="${DIAMETER:-10}"
NEUCOEFF="${NEUCOEFF:-0}"
CELLPROB=""
BASE_RUN="${BASE_RUN:-1}"
COLOR_UP="${COLOR_UP:-#ff4b00}"
COLOR_DOWN="${COLOR_DOWN:-#005aff}"
REP_ROI=""
GLOB=""
DATE=""
TAKE_ALL=0

usage() { sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    -d|--dataset)   WANTED+=("$2"); shift 2;;
    --glob)         GLOB="$2"; shift 2;;
    --date)         DATE="$2"; shift 2;;
    --all)          TAKE_ALL=1; shift;;
    -s|--stages)    STAGES="$2"; shift 2;;
    -p|--pattern)   PATTERN="$2"; shift 2;;
    --ws)           WS="$2"; shift 2;;
    --device)       DEVICE="$2"; shift 2;;
    --diameter)     DIAMETER="$2"; shift 2;;
    --neucoeff)     NEUCOEFF="$2"; shift 2;;
    --cellprob)     CELLPROB="$2"; shift 2;;
    --base-run)     BASE_RUN="$2"; shift 2;;
    --roi)          REP_ROI="$2"; shift 2;;
    -n|--dry-run)   DRYRUN=1; shift;;
    -h|--help)      usage;;
    *) echo "unknown option: $1" >&2; usage;;
  esac
done

# --- resolve the list of datasets -------------------------------------------
declare -a DATASETS=()
add_ds() {
  local d="$1"
  # a bare name is taken relative to the workspace, a path is used as given
  [ -d "$d" ] || d="$WS/${d%/}"
  d="${d%/}"
  if [ ! -d "$d/raw" ]; then
    echo "  [skip] no raw/ under $d" >&2
    return 1
  fi
  DATASETS+=("$d")
}

for w in "${WANTED[@]:-}"; do [ -n "$w" ] && add_ds "$w"; done
if [ -n "$GLOB" ]; then
  for d in "$WS"/$GLOB; do [ -d "$d" ] && add_ds "$d"; done
fi
if [ -n "$DATE" ]; then
  for d in "$WS/${DATE}"_*; do [ -d "$d" ] && add_ds "$d"; done
fi
if [ "$TAKE_ALL" -eq 1 ]; then
  for d in "$WS"/*; do [ -d "$d" ] && add_ds "$d"; done
fi

if [ "${#DATASETS[@]}" -eq 0 ]; then
  echo "ERROR: no datasets. Give -d, --glob, --date or --all." >&2
  echo "available under $WS:" >&2
  ls -1 "$WS" 2>/dev/null | sed 's/^/  /' >&2
  exit 2
fi

has() { case ",$STAGES," in *",$1,"*|*",all,"*) return 0;; *) return 1;; esac; }

echo "workspace : $WS"
echo "datasets  : ${#DATASETS[@]}"
for d in "${DATASETS[@]}"; do echo "   $(basename "$d")"; done
echo "stages    : $STAGES"
[ "$DRYRUN" -eq 1 ] && echo "MODE      : dry run"
echo

declare -a RESULT_NAME=() RESULT_STATE=() RESULT_TIME=()

# --- one dataset -------------------------------------------------------------
process_one() {
  local DATASET="$1"
  local SUB W S2P LEDGER AUC pat
  SUB="$(basename "$DATASET" | sed -n 's/.*_\(sub-[^_]*\)_.*/\1/p')"
  W="$DATASET/work"
  S2P="$W/s2p_series/suite2p/plane0"
  LEDGER="$W/s2p_series/frame_ledger.csv"
  AUC="$W/eauc"
  FAILED=""

  pat="$PATTERN"
  if [ -z "$pat" ]; then
    pat="${SUB}_*_cond-*_run-*.tif"
    ls "$DATASET/raw/"$pat >/dev/null 2>&1 || pat="*_cond-*_run-*.tif"
  fi

  echo "=============================================================="
  echo "$(basename "$DATASET")   subject ${SUB:-unknown}   pattern $pat"
  echo "=============================================================="

  run() {  # run <env> <label> <args...>
    local env="$1" label="$2"; shift 2
    echo "--- [$label] (env: $env)"
    if [ "$DRYRUN" -eq 1 ]; then
      printf '    conda run -n %s python' "$env"; printf ' %q' "$@"; echo
      return 0
    fi
    if ! conda run -n "$env" --no-capture-output python "$@"; then
      echo "!!! [$label] failed" >&2
      FAILED="$FAILED$label "
      return 1
    fi
  }

  if has meta; then
    run "$ENV_META" meta "$IVWIB/scripts/generate_metadata.py" "$DATASET/"
  fi

  if has detect; then
    local -a args
    args=("$SCRIPTS/run_suite2p_series.py" --dataset "$DATASET"
          --pattern "$pat" --functional-chan 1 --torch-device "$DEVICE"
          --smooth-sigma-time 0 --algorithm cellpose --cellpose-img meanImg
          --diameter "$DIAMETER" --save-path "$W/s2p_series")
    [ -n "$CELLPROB" ] && args+=(--cellprob-threshold "$CELLPROB")
    run "$ENV_S2P" detect "${args[@]}"
  fi

  # everything downstream reads the traces, so say so rather than failing
  # stage by stage with the same missing file
  if [ "$DRYRUN" -eq 0 ] && [ ! -f "$S2P/F.npy" ] \
     && { has qc || has auc || has figs || has spikes || has movie \
          || has cnmf || has deepcad || has foopsi || has compare; }; then
    echo "!!! no traces at $S2P — the detect stage has not run" >&2
    FAILED="${FAILED}no-detection "
    return 1
  fi

  if has qc; then
    run "$ENV_S2P" qc "$SCRIPTS/qc_feasibility.py" --s2p-dir "$S2P" \
        --dataset "$DATASET" --all-roi --neucoeff "$NEUCOEFF" --out "$W/qc"
    run "$ENV_S2P" "mean per run" "$SCRIPTS/fig_mean_per_run.py" \
        --s2p-dir "$S2P" --ledger "$LEDGER" --dataset "$DATASET" \
        --out "$W/mean_per_run" --rois
  fi

  if has auc; then
    run "$ENV_S2P" auc "$SCRIPTS/run_event_auc.py" --s2p-dir "$S2P" \
        --dataset "$DATASET" --ledger "$LEDGER" --all-roi \
        --neucoeff "$NEUCOEFF" --smooth matched --fp-method cumulative \
        --out "$AUC"
  fi

  if has figs; then
    run "$ENV_S2P" "auc lines" "$SCRIPTS/fig_auc_lines.py" \
        --csv "$AUC/auc_per_roi_per_run.csv" --out "$W/fig_auc_lines"
    run "$ENV_S2P" "change pie" "$SCRIPTS/fig_change_pie.py" \
        --auc-dir "$AUC" --ledger "$LEDGER" --dataset "$DATASET" \
        --from-run "$BASE_RUN" --to-run all \
        --color-up "$COLOR_UP" --color-down "$COLOR_DOWN" \
        --label "$(basename "$DATASET")" --out "$W/fig_change_pie"
    run "$ENV_S2P" "auc area" "$SCRIPTS/fig_auc_area.py" \
        --s2p-dir "$S2P" --labels raw --dataset "$DATASET" --ledger "$LEDGER" \
        --all-roi --neucoeff "$NEUCOEFF" --out "$W/fig_auc_area"
    local -a rep
    rep=("$SCRIPTS/fig_representative.py" --s2p-dir "$S2P" --dataset "$DATASET"
         --ledger "$LEDGER" --all-roi --neucoeff "$NEUCOEFF"
         --auc-csv "$AUC/auc_per_roi_per_run.csv"
         --events-csv "$AUC/events.csv" --out "$W/fig_representative")
    [ -n "$REP_ROI" ] && rep+=(--roi $REP_ROI)
    run "$ENV_S2P" representative "${rep[@]}"
  fi

  if has spikes; then
    run "$ENV_S2P" spikes "$SCRIPTS/run_suite2p_spikes.py" --s2p-dir "$S2P" \
        --dataset "$DATASET" --ledger "$LEDGER" --all-roi \
        --neucoeff "$NEUCOEFF" --tau 0.27 --out "$W/spks"
    run "$ENV_S2P" "spikes per run" "$SCRIPTS/fig_spikes_per_run.py" \
        --s2p-dir "$W/spks/suite2p/plane0" --ledger "$LEDGER" \
        --dataset "$DATASET" --all-roi --out "$W/spikes_per_run"
    run "$ENV_S2P" "spike lines" "$SCRIPTS/fig_auc_lines.py" \
        --value spike_rate_per_min --csv "$W/spks/auc_per_roi_per_run.csv" \
        --out "$W/fig_spike_lines"
    run "$ENV_S2P" "spike pie" "$SCRIPTS/fig_change_pie.py" \
        --auc-dir "$W/spks" --ledger "$W/spks/frame_ledger.csv" \
        --dataset "$DATASET" --value spike_rate_per_min \
        --from-run "$BASE_RUN" --to-run all \
        --color-up "$COLOR_UP" --color-down "$COLOR_DOWN" \
        --label "$(basename "$DATASET") spikes" --out "$W/fig_spike_pie"
  fi

  if has movie; then
    mkdir -p "$DATASET/results"
    run "$ENV_S2P" movie "$SCRIPTS/make_movie.py" --s2p-dir "$S2P" \
        --dataset "$DATASET" --ledger "$LEDGER" --rois --all-roi \
        --speed 30 --scale 4 --grid \
        --out "$DATASET/results/${SUB}_registered.avi"
  fi

  if has cnmf || has deepcad; then
    if [ ! -f "$W/registered.tif" ]; then
      run "$ENV_S2P" export "$SCRIPTS/extract_from_movie.py" --s2p-dir "$S2P" \
          --export "$W/registered.tif" --crop
    else
      echo "--- [export] registered.tif already there, keeping it"
    fi
  fi

  if has cnmf; then
    if [ ! -f "$W/registered_cnmf.tif" ]; then
      run "$ENV_CAIMAN" cnmf "$SCRIPTS/run_caiman_denoise.py" \
          --input "$W/registered.tif" --out "$W/registered_cnmf.tif" \
          --p 0 --gSig 4 --K 40 --fs 13.29
    fi
    run "$ENV_S2P" "cnmf extract" "$SCRIPTS/extract_from_movie.py" \
        --s2p-dir "$S2P" --movie "$W/registered_cnmf.tif" \
        --out "$W/s2p_cnmf/suite2p/plane0"
    run "$ENV_S2P" "cnmf auc" "$SCRIPTS/run_event_auc.py" \
        --s2p-dir "$W/s2p_cnmf/suite2p/plane0" --dataset "$DATASET" \
        --ledger "$W/s2p_cnmf/frame_ledger.csv" --all-roi \
        --neucoeff "$NEUCOEFF" --smooth matched --fp-method cumulative \
        --out "$W/eauc_cnmf"
  fi

  if has deepcad; then
    if [ ! -f "$W/registered_deepcad.tif" ]; then
      echo "--- [deepcad] (env: $ENV_DEEPCAD)"
      if [ "$DRYRUN" -eq 1 ]; then
        echo "    conda run -n $ENV_DEEPCAD python $IVWIB/scripts/deepcad_run.py --input $W/registered.tif ..."
      elif ! conda run -n "$ENV_DEEPCAD" --no-capture-output python \
             "$IVWIB/scripts/deepcad_run.py" --input "$W/registered.tif" \
             --out "$W/registered_deepcad.tif" --work-dir "$W/deepcad" \
             --epochs 10 --gpu 0 --keep-rescale; then
        echo "!!! [deepcad] failed" >&2
        FAILED="${FAILED}deepcad "
      fi
    fi
    if [ "$DRYRUN" -eq 1 ] || [ -f "$W/registered_deepcad.tif" ]; then
      run "$ENV_S2P" "deepcad extract" "$SCRIPTS/extract_from_movie.py" \
          --s2p-dir "$S2P" --movie "$W/registered_deepcad.tif" \
          --out "$W/s2p_deepcad/suite2p/plane0"
      run "$ENV_S2P" "deepcad auc" "$SCRIPTS/run_event_auc.py" \
          --s2p-dir "$W/s2p_deepcad/suite2p/plane0" --dataset "$DATASET" \
          --ledger "$W/s2p_deepcad/frame_ledger.csv" --all-roi \
          --neucoeff "$NEUCOEFF" --smooth matched --fp-method cumulative \
          --out "$W/eauc_deepcad"
    fi
  fi

  if has foopsi; then
    run "$ENV_CAIMAN" foopsi "$SCRIPTS/run_caiman_traces.py" --s2p-dir "$S2P" \
        --dataset "$DATASET" --ledger "$LEDGER" --all-roi \
        --neucoeff "$NEUCOEFF" --p 1 --g fixed --tau 0.27 --out "$W/foopsi"
  fi

  if has compare; then
    local -a csvs labels planes plabels
    csvs=(); labels=(); planes=(); plabels=()
    [ -f "$AUC/auc_per_roi_per_run.csv" ] && { csvs+=("$AUC/auc_per_roi_per_run.csv"); labels+=(raw); }
    [ -f "$W/eauc_cnmf/auc_per_roi_per_run.csv" ] && { csvs+=("$W/eauc_cnmf/auc_per_roi_per_run.csv"); labels+=(CNMF); }
    [ -f "$W/eauc_deepcad/auc_per_roi_per_run.csv" ] && { csvs+=("$W/eauc_deepcad/auc_per_roi_per_run.csv"); labels+=(DeepCAD); }
    [ -f "$W/foopsi/auc_per_roi_per_run.csv" ] && { csvs+=("$W/foopsi/auc_per_roi_per_run.csv"); labels+=(foopsi); }
    if [ "${#csvs[@]}" -ge 2 ]; then
      run "$ENV_S2P" compare "$SCRIPTS/fig_auc_lines.py" \
          --labels "${labels[@]}" --csv "${csvs[@]}" --out "$W/fig_compare"
      planes=("$S2P"); plabels=(raw)
      [ -f "$W/s2p_cnmf/suite2p/plane0/F.npy" ] && { planes+=("$W/s2p_cnmf/suite2p/plane0"); plabels+=(CNMF); }
      [ -f "$W/s2p_deepcad/suite2p/plane0/F.npy" ] && { planes+=("$W/s2p_deepcad/suite2p/plane0"); plabels+=(DeepCAD); }
      [ "${#planes[@]}" -ge 2 ] && run "$ENV_S2P" "area compare" \
          "$SCRIPTS/fig_auc_area.py" --s2p-dir "${planes[@]}" \
          --labels "${plabels[@]}" --dataset "$DATASET" --ledger "$LEDGER" \
          --all-roi --neucoeff "$NEUCOEFF" --out "$W/fig_area_compare"
    else
      echo "--- [compare] fewer than two variants present, skipping"
    fi
  fi

  [ -z "$FAILED" ] || return 1
  return 0
}

# --- loop ---------------------------------------------------------------------
t_all=$SECONDS
for DS in "${DATASETS[@]}"; do
  t0=$SECONDS
  FAILED=""
  if process_one "$DS"; then
    state="ok"
  else
    state="FAILED: ${FAILED:-see above}"
  fi
  RESULT_NAME+=("$(basename "$DS")")
  RESULT_STATE+=("$state")
  RESULT_TIME+=("$(( SECONDS - t0 ))")
  echo
done

echo "=============================================================="
printf '%-34s %8s  %s\n' dataset seconds state
n_bad=0
for i in "${!RESULT_NAME[@]}"; do
  printf '%-34s %8s  %s\n' "${RESULT_NAME[$i]}" "${RESULT_TIME[$i]}" "${RESULT_STATE[$i]}"
  [ "${RESULT_STATE[$i]}" = "ok" ] || n_bad=$(( n_bad + 1 ))
done
echo "total $(( (SECONDS - t_all) / 60 )) min $(( (SECONDS - t_all) % 60 )) s"
[ "$n_bad" -eq 0 ] || { echo "$n_bad of ${#RESULT_NAME[@]} dataset(s) had failures" >&2; exit 1; }
