#!/usr/bin/env python
"""Denoise a movie with CaImAn's CNMF and write the result as a TIFF.

CNMF factorises the movie into spatial footprints and their time courses plus a
low-rank background, and reconstructing from that factorisation is a denoiser:
everything that does not fit a small number of spatial components is discarded.
It is the strongest denoiser available here because it pools pixels with
optimised weights rather than smoothing in time.

It also carries an assumption the other denoisers do not. Each component's time
course is constrained to be an autoregressive process driven by non-negative
spikes, so the reconstruction contains transients whose SHAPE was imposed by a
calcium model rather than measured. That is deconvolution, reintroduced through
the reconstruction. Two consequences:

  - The output will look like clean calcium traces whether or not the input
    contained any, because that is what the model can represent. Judging it by
    eye is therefore not informative.
  - It must pass deepcad_gate_ca.py before any number from it is reported,
    exactly like any other denoiser, and for the same reasons.

Writing --p 0 disables the autoregressive constraint, which removes the imposed
shape at the cost of some denoising. That version is the honest one to compare
against.

Example
-------
    python run_caiman_denoise.py --input work/registered.tif \
        --out work/registered_cnmf.tif --p 0 --K 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="CNMF denoise, written back as a movie",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="registered TIFF")
    p.add_argument("--out", type=Path, required=True, help="denoised TIFF")
    p.add_argument("--fs", type=float, default=13.29)
    p.add_argument("--tau", type=float, default=0.27,
                   help="indicator decay, used for the AR model when --p > 0")
    p.add_argument("--K", type=int, default=40,
                   help="components per patch; set well above the expected "
                        "cell count so no cell is forced to share one")
    p.add_argument("--gSig", type=int, default=4,
                   help="half-width of a cell in pixels")
    p.add_argument("--p", type=int, default=0, choices=[0, 1, 2],
                   help="autoregressive order. 0 leaves the time courses "
                        "unconstrained, so the output keeps whatever shape the "
                        "data had; 1 or 2 impose calcium dynamics and the "
                        "result is a deconvolution, not only a denoising")
    p.add_argument("--rf", type=int, default=None,
                   help="patch half-size; None processes the whole field at once")
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--merge-thr", type=float, default=0.85)
    p.add_argument("--gnb", type=int, default=2, help="background components")
    p.add_argument("--no-background", action="store_true",
                   help="reconstruct without the background term, leaving only "
                        "the component signal")
    p.add_argument("--no-refit", dest="refit", action="store_false",
                   help="skip the second CNMF pass")
    p.add_argument("--n-processes", type=int, default=None)
    p.add_argument("--dtype", default="float32", choices=["float32", "int16"])
    args = p.parse_args(argv)

    try:
        import caiman as cm
        from caiman.source_extraction.cnmf import cnmf as cnmf_mod
        from caiman.source_extraction.cnmf import params as cnmf_params
    except ImportError:
        print("ERROR: caiman is required. Use the caiman environment.",
              file=sys.stderr)
        return 2
    import tifffile

    src = args.input.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent

    with tifffile.TiffFile(src) as tf:
        n_t = len(tf.pages)
        ny, nx = tf.pages[0].shape[:2]
    print(f"input: {src.name}  {n_t} frames  {ny} x {nx}")
    print(f"CNMF: K={args.K}  gSig={args.gSig}  p={args.p}  "
          f"{'patches rf=%d' % args.rf if args.rf else 'whole field'}")
    if args.p > 0:
        print("  p > 0 imposes calcium dynamics on every time course; the "
              "output is a\n  deconvolution and must clear the gate before any "
              "number from it is used")

    dview = None
    cwd = os.getcwd()
    try:
        _, dview, n_proc = cm.cluster.setup_cluster(
            backend="multiprocessing", n_processes=args.n_processes,
            single_thread=False)
        print(f"cluster: {n_proc} processes")
        os.chdir(work)

        fname_map = cm.save_memmap([str(src)], base_name="cnmf_", order="C",
                                   dview=dview)
        Yr, dims, T = cm.load_memmap(fname_map)
        images = Yr.T.reshape((T,) + dims, order="F")

        opts = cnmf_params.CNMFParams(params_dict={
            "data": {"fnames": [str(src)], "fr": args.fs,
                     "decay_time": args.tau},
            "init": {"K": args.K, "gSig": (args.gSig, args.gSig),
                     "nb": args.gnb, "method_init": "greedy_roi"},
            "patch": {"rf": args.rf, "stride": args.stride,
                      "only_init": False, "nb_patch": args.gnb},
            "preprocess": {"p": args.p},
            "temporal": {"p": args.p},
            "merging": {"merge_thr": args.merge_thr},
        })
        cnm = cnmf_mod.CNMF(n_proc, params=opts, dview=dview)
        # Some CaImAn versions return the object from fit(), others update it in
        # place and return None. Keep whichever object actually holds estimates
        # rather than assuming either convention.
        res = cnm.fit(images)
        cnm = res if getattr(res, "estimates", None) is not None else cnm
        if getattr(cnm, "estimates", None) is None:
            print("ERROR: CNMF produced no estimates. With K components and "
                  "gSig too large for the\n  field, initialisation can find "
                  "nothing; try a smaller --gSig or --K.", file=sys.stderr)
            return 2
        print(f"  {cnm.estimates.A.shape[1]} components, "
              f"{cnm.estimates.b.shape[1]} background")
        if args.refit:
            res = cnm.refit(images, dview=dview)
            cnm = res if getattr(res, "estimates", None) is not None else cnm
            print(f"  after refit: {cnm.estimates.A.shape[1]} components")

        est = cnm.estimates
        if est.C is None or est.A is None:
            print("ERROR: the factorisation has no components to reconstruct "
                  "from.", file=sys.stderr)
            return 2
        rec = est.A.dot(est.C)
        if not args.no_background:
            rec = rec + est.b.dot(est.f)
        mov_out = np.asarray(rec).T.reshape((T,) + dims, order="F")
    finally:
        os.chdir(cwd)
        if dview is not None:
            try:
                cm.stop_server(dview=dview)
            except Exception:  # noqa: BLE001
                pass

    if args.dtype == "int16":
        mov_out = np.clip(mov_out, -32768, 32767)
    tifffile.imwrite(out, mov_out.astype(args.dtype))
    print(f"\nwrote {out}  {mov_out.shape} {args.dtype} "
          f"({out.stat().st_size / 1e9:.2f} GB)")

    with open(out.with_suffix(".json"), "w") as fh:
        json.dump({"input": str(src), "n_components": int(est.A.shape[1]),
                   "n_background": int(est.b.shape[1]), "p": args.p,
                   "K": args.K, "gSig": args.gSig, "rf": args.rf,
                   "background_included": not args.no_background,
                   "note": "reconstruction from the CNMF factorisation. With "
                           "p > 0 the time courses are autoregressive, so the "
                           "transient shape is imposed by the model."},
                  fh, indent=2)
    print("\nnext:")
    print("  1. gate it:   python deepcad_gate_ca.py --make-synth work/gate_ca.tif")
    print("                python run_caiman_denoise.py --input work/gate_ca.tif "
          "--out work/gate_ca_cnmf.tif --p 0")
    print("                python deepcad_gate_ca.py --eval work/gate_ca.tif "
          "--denoised work/gate_ca_cnmf.tif")
    print("  2. re-extract through the same ROIs, then run the usual analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
