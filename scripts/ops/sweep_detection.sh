#!/usr/bin/env bash
# sweep_detection.sh -- run every detection setting worth trying on one dataset.
#
# The pilot found ROIs of ~3.2 px equivalent diameter where a soma is ~7.6 px
# (45 px area) at 1.96 um/px. That is a sixth of cell-body size, so the ROIs are
# fragments of neuropil and processes, not somata. Threshold tuning does not fix
# that -- the detection SCALE does -- so this sweep varies scale first and
# thresholds only within a fixed scale.
#
# Each run writes to its own --save-path, so nothing is overwritten and every
# combination stays available for comparison. Compare with compare_detection.py.
#
# Usage:
#   bash sweep_detection.sh <dataset_dir> [pattern] [device]
#
# Example:
#   bash sweep_detection.sh \
#     /media/tshino/DATA/workspace/2026_aqp4ko_water_intoxication/2photon/20260803_sub-sk28_ses-01 \
#     'sk28_ch0_*_MC.tif' cuda

set -uo pipefail

D="${1:?usage: sweep_detection.sh <dataset_dir> [pattern] [device]}"
PAT="${2:-*_ch0_*_MC.tif}"
DEV="${3:-cuda}"
SCRIPT="${SCRIPT:-run_roi_suite2p.py}"

run() {                      # run <tag> <extra args...>
  local tag="$1"; shift
  local out="$D/work/s2p_$tag"
  if [ -d "$out/suite2p/plane0" ]; then
    echo "=== $tag  [exists, skipping] ==="
    return 0
  fi
  echo "=== $tag ==="
  python "$SCRIPT" --dataset "$D" --pattern "$PAT" --torch-device "$DEV" \
      --save-path "$out" "$@" 2>&1 \
    | grep -E "detected|ROI size|npix|These ROIs|spatial_scale|cellpose img|ERROR" \
    || echo "  (run failed; see full output by running the command directly)"
  echo
}

echo "dataset : $D"
echo "pattern : $PAT"
echo "device  : $DEV"
echo

# --------------------------------------------------------------------------
# A. functional detection (sparsery), varying the spatial scale.
#    scale 0=auto, 1=6px, 2=12px, 3=24px, 4=48px. Auto failed on this data and
#    fell back to 1, so scales are set explicitly here.
# --------------------------------------------------------------------------
run func_ss0 --anatomical-only 0 --spatial-scale 0
run func_ss1 --anatomical-only 0 --spatial-scale 1
run func_ss2 --anatomical-only 0 --spatial-scale 2
run func_ss3 --anatomical-only 0 --spatial-scale 3
run func_ss4 --anatomical-only 0 --spatial-scale 4

# --------------------------------------------------------------------------
# B. best scale, with size and overlap filters to drop fragments.
#    Change SS below once A shows which scale gives cell-body-sized ROIs.
# --------------------------------------------------------------------------
SS="${SS:-2}"
run func_ss${SS}_npix05  --anatomical-only 0 --spatial-scale "$SS" --npix-norm-min 0.5
run func_ss${SS}_npix10  --anatomical-only 0 --spatial-scale "$SS" --npix-norm-min 1.0
run func_ss${SS}_ovl50   --anatomical-only 0 --spatial-scale "$SS" --max-overlap 0.5

# --------------------------------------------------------------------------
# C. best scale, varying the detection threshold. Lower finds more, including
#    more false positives; higher keeps only strong candidates.
# --------------------------------------------------------------------------
run func_ss${SS}_ts05 --anatomical-only 0 --spatial-scale "$SS" --threshold-scaling 0.5
run func_ss${SS}_ts20 --anatomical-only 0 --spatial-scale "$SS" --threshold-scaling 2.0

# --------------------------------------------------------------------------
# D. keep processes attached instead of cropping to the soma. Useful as a
#    contrast: if ROIs grow a lot, the detector was finding real structure and
#    soma_crop was trimming it; if not, it was finding fragments.
# --------------------------------------------------------------------------
run func_ss${SS}_nocrop --anatomical-only 0 --spatial-scale "$SS" --no-soma-crop

# --------------------------------------------------------------------------
# E. anatomical detection (Cellpose), varying the image and the diameter.
#    meanImg is activity-independent, which matters for a design that compares
#    activity between conditions; max_proj shows active somata far more clearly
#    but weights detection by activity, so it is diagnostic here rather than a
#    candidate for the main analysis.
# --------------------------------------------------------------------------
run cp_mean_d8   --anatomical-only 2 --cellpose-img meanImg  --diameter 8
run cp_mean_d10  --anatomical-only 2 --cellpose-img meanImg  --diameter 10
run cp_mean_d12  --anatomical-only 2 --cellpose-img meanImg  --diameter 12
run cp_max_d8    --anatomical-only 2 --cellpose-img max_proj --diameter 8
run cp_max_d10   --anatomical-only 2 --cellpose-img max_proj --diameter 10
run cp_max_d12   --anatomical-only 2 --cellpose-img max_proj --diameter 12
run cp_auto_d0   --anatomical-only 2 --cellpose-img "max_proj / meanImg" --diameter 0

echo "=== sweep complete ==="
echo "compare with:  python compare_detection.py $D"
