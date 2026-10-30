"""Command-line interface for MRC maps: ``phaselp filter``, ``phaselp compare`` and ``phaselp fsc``."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np

from .compare import compare_maps, local_rms
from .filters import amplitude_only_filter, ramped_phase_filter
from .fsc import fsc
from .maps import read_map, write_map


def _save_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    np.savetxt(path, np.column_stack(list(columns.values())), delimiter=",",
               header=",".join(columns), comments="", fmt="%.6g")


def _cmd_filter(args) -> None:
    m = read_map(args.input)
    if args.amplitude_only:
        out = amplitude_only_filter(m.data, m.voxel_size, args.target_res, args.parent_res, kappa=args.kappa)
    else:
        out = ramped_phase_filter(m.data, m.voxel_size, args.target_res, args.parent_res,
                                  kappa=args.kappa, seed=args.seed)
    write_map(args.output, out, m.voxel_size, m.origin)


def _cmd_fsc(args) -> None:
    a, b = read_map(args.map1), read_map(args.map2)
    if a.data.shape != b.data.shape or not np.isclose(a.voxel_size, b.voxel_size):
        raise SystemExit("maps must share grid and voxel size; use `phaselp compare` to register them")
    frequency, values = fsc(a.data, b.data, a.voxel_size)
    if args.out:
        _save_csv(args.out, {"frequency_invA": frequency, "fsc": values})
    if args.plot:
        from .plotting import plot_fsc

        plot_fsc({"frequency": frequency, "FSC": values}).savefig(args.plot, dpi=200)
    if not args.out and not args.plot:
        for f, v in zip(frequency, values):
            print(f"{f:.5f}\t{v:.4f}")


def _cmd_compare(args) -> None:
    parent, experimental = read_map(args.parent), read_map(args.experimental)
    seeds = list(range(args.seed, args.seed + args.seeds))
    result = compare_maps(
        parent.data, experimental.data,
        parent_voxel_size=parent.voxel_size, voxel_size=experimental.voxel_size,
        target_res=args.target_res, parent_res=args.parent_res, seeds=seeds, kappa=args.kappa,
        register_maps=not args.no_register, translation_bound=args.translation_bound,
    )
    out, voxel, origin = Path(args.out), experimental.voxel_size, experimental.origin
    out.mkdir(parents=True, exist_ok=True)
    write_map(out / "parent_registered.mrc", result.parent, voxel, origin)
    write_map(out / "experimental_standardized.mrc", result.experimental, voxel, origin)
    write_map(out / "amplitude_only.mrc", result.amplitude_only, voxel, origin)
    for seed, filtered in zip(result.seeds, result.filtered):
        write_map(out / f"ramped_phase_seed{seed}.mrc", filtered, voxel, origin)
    difference = result.differences()[0]
    write_map(out / f"difference_seed{result.seeds[0]}.mrc", difference, voxel, origin)
    write_map(out / f"local_rms_seed{result.seeds[0]}.mrc",
              local_rms(difference, args.target_res, voxel), voxel, origin)

    curves = result.fsc_curves()
    with warnings.catch_warnings():  # the DC shell is NaN for every seed
        warnings.simplefilter("ignore", RuntimeWarning)
        mean, sd = np.nanmean(curves["ramped_phase"], axis=0), np.nanstd(curves["ramped_phase"], axis=0)
    table = {"frequency_invA": curves["frequency"], "expected_c": curves["expected"],
             "fsc_ramped_phase_mean": mean, "fsc_ramped_phase_sd": sd,
             "fsc_amplitude_only": curves["amplitude_only"],
             "fsc_experimental": curves["experimental"]}
    _save_csv(out / "fsc_with_parent.csv", table)
    record = {"target_res": result.target_res, "parent_res": result.parent_res,
              "kappa": result.kappa, "seeds": list(result.seeds), "voxel_size": voxel}
    if result.registration is not None:
        reg = result.registration
        record["registration"] = {
            "scale": reg.scale, "rotation_degrees": reg.rotation_degrees,
            "translation_zyx_voxels": reg.translation_voxels.tolist(),
            "parameters": reg.parameters.tolist(), "ncc_before": reg.score_before,
            "ncc_after": reg.score_after, "message": reg.message,
        }
    (out / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")

    if args.plots:
        from .plotting import plot_differences, plot_fsc, plot_sections

        plot_sections([result.parent, result.experimental, result.amplitude_only, result.filtered[0]],
                      ["parent (registered)", "experimental", "amplitude only", "ramped phase"]
                      ).savefig(out / "sections.png", dpi=200)
        plot_fsc(curves, target_res=args.target_res).savefig(out / "fsc_with_parent.png", dpi=200)
        plot_differences({"filtered − experimental": difference}, sigma=args.target_res,
                         voxel_size=voxel).savefig(out / "difference.png", dpi=200)
    print(json.dumps(record, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phaselp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def filter_options(p):
        p.add_argument("--target-res", type=float, required=True, help="target resolution R_T (Å)")
        p.add_argument("--parent-res", type=float, default=None,
                       help="parent resolution R_H (Å) for equation (7); omit to treat the parent as ideal")
        p.add_argument("--kappa", type=float, default=100.0, help="SSNR scale of equation (5)")
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("filter", help="ramped-phase (or amplitude-only) low-pass filter of one map")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    filter_options(p)
    p.add_argument("--amplitude-only", action="store_true", help="no phase offsets (control)")
    p.set_defaults(func=_cmd_filter)

    p = sub.add_parser("compare", help="register, filter and amplitude-match a parent to an experimental map")
    p.add_argument("parent", type=Path, help="higher-resolution map")
    p.add_argument("experimental", type=Path, help="lower-resolution map; defines the output grid")
    p.add_argument("out", type=Path, help="output directory")
    filter_options(p)
    p.add_argument("--seeds", type=int, default=1, help="number of ramped-phase maps")
    p.add_argument("--no-register", action="store_true", help="maps are already aligned")
    p.add_argument("--translation-bound", type=float, default=20.0, help="voxels")
    p.add_argument("--plots", action="store_true", help="also write PNG figures")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("fsc", help="FSC between two maps on the same grid")
    p.add_argument("map1", type=Path)
    p.add_argument("map2", type=Path)
    p.add_argument("--out", type=Path, help="CSV output")
    p.add_argument("--plot", type=Path, help="PNG output")
    p.set_defaults(func=_cmd_fsc)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
