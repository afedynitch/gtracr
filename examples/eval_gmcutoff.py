import sys
import os
import numpy as np
import pickle
import argparse

from tqdm import tqdm
from gtracr.geomagnetic_cutoffs import GMRC
from gtracr.utils import location_dict
from gtracr.plotting import plot_gmrc_scatter, plot_gmrc_heatmap

# add filepath of gtracr to sys.path
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
PLOT_DIR = os.path.join(PARENT_DIR, "..", "gtracr_plots")

# create directory if gtracr_plots dir does not exist
if not os.path.isdir(PLOT_DIR):
    os.mkdir(PLOT_DIR)


def export_as_pkl(fpath, ds):
    with open(fpath, "wb") as f:
        pickle.dump(ds, f, protocol=-1)


def _evaluate_gmrc_table(gmrc, dt=1e-5, max_time=1.):
    """
    Sequential GMRC evaluation using the C++ TrajectoryTracer with the
    precomputed 3-D IGRF lookup table (bfield_type='t').

    One CppTrajectoryTracer is built (table generation happens in its C++
    constructor) and reused for every direction and rigidity via reset().
    This avoids rebuilding the table for each of the 10 000 MC samples.
    Runs in the main process (no pickling of the 24 MB table into workers).
    """
    from gtracr.trajectory import Trajectory
    from gtracr.lib._libgtracr import TrajectoryTracer as CppTrajectoryTracer
    from gtracr.lib.constants import ELEMENTARY_CHARGE, KG_PER_GEVC2, KG_M_S_PER_GEVC

    rigidity_list = list(gmrc.rigidity_list)
    max_step = int(np.ceil(max_time / dt))

    # Build a reference Trajectory just to get charge/mass/start_alt/igrf_params.
    ref_traj = Trajectory(
        plabel=gmrc.plabel,
        location_name=gmrc.location,
        zenith_angle=45., azimuth_angle=0.,
        particle_altitude=gmrc.palt,
        rigidity=rigidity_list[0],
        bfield_type="igrf",
        date=gmrc.date,
    )
    charge_si = ref_traj.charge * ELEMENTARY_CHARGE
    mass_si   = ref_traj.mass   * KG_PER_GEVC2

    print("Building IGRF lookup table in C++ (64×128×256 grid)…", flush=True)
    # bfield_type='t' triggers table generation inside the C++ constructor.
    tracer = CppTrajectoryTracer(
        charge_si, mass_si,
        ref_traj.start_alt, ref_traj.esc_alt,
        dt, max_step,
        't', ref_traj.igrf_params,
        gmrc.solver_char, gmrc.atol, gmrc.rtol,
    )
    print("Table built. Running MC loop…", flush=True)

    rng = np.random.default_rng()

    for i in tqdm(range(gmrc.iter_num), desc="GMRC (table)"):
        azimuth, zenith = rng.random(2) * np.array([360., 180.])

        traj = Trajectory(
            plabel=gmrc.plabel,
            location_name=gmrc.location,
            zenith_angle=zenith,
            azimuth_angle=azimuth,
            particle_altitude=gmrc.palt,
            rigidity=rigidity_list[0],
            bfield_type="igrf",   # only used for coordinate transform, not field eval
            date=gmrc.date,
        )

        ref_mom_si = traj.particle.momentum * KG_M_S_PER_GEVC
        pos      = traj.particle_sixvector[:3]
        mom_unit = traj.particle_sixvector[3:] / ref_mom_si

        # For zenith > 90, detector_to_geocentric reduces start_alt via
        # cos²(zenith). Update the shared tracer's termination boundary to
        # match, so the trajectory is not terminated prematurely.
        tracer.set_start_altitude(traj.start_alt)

        rcutoff = 0.0
        for rigidity in rigidity_list:
            traj.particle.set_from_rigidity(rigidity)
            mom_si = traj.particle.momentum * KG_M_S_PER_GEVC
            vec0 = list(pos) + list(mom_unit * mom_si)
            tracer.reset()
            tracer.evaluate(0.0, vec0)
            if tracer.particle_escaped:
                rcutoff = rigidity
                break

        gmrc.data_dict["azimuth"][i] = azimuth
        gmrc.data_dict["zenith"][i]  = zenith
        gmrc.data_dict["rcutoff"][i] = rcutoff


def _run_gmrc(gmrc, args):
    """Evaluate gmrc using the field mode specified in args, then plot."""
    plabel = "p+"
    ngrid_azimuth = 360
    ngrid_zenith = 180
    locname = gmrc.location

    if args.field_mode == "table":
        _evaluate_gmrc_table(gmrc)
    else:
        gmrc.evaluate()

    plot_gmrc_scatter(gmrc.data_dict,
                      locname,
                      plabel,
                      bfield_type=args.bfield_type,
                      iter_num=args.iter_num,
                      show_plot=args.show_plot)

    interpd_gmrc_data = gmrc.interpolate_results(
        ngrid_azimuth=ngrid_azimuth,
        ngrid_zenith=ngrid_zenith,
    )

    plot_gmrc_heatmap(interpd_gmrc_data,
                      gmrc.rigidity_list,
                      locname=locname,
                      plabel=plabel,
                      bfield_type=args.bfield_type,
                      show_plot=args.show_plot)


def eval_gmrc(args):
    # create particle trajectory with desired particle and energy
    plabel = "p+"
    particle_altitude = 100.

    # change initial parameters if debug mode is set
    if args.debug_mode:
        args.iter_num = 10
        args.show_plot = True

    if args.field_mode == "table" and args.iter_num > 1000:
        print(f"Note: --field-mode table runs sequentially (table cannot be shared "
              f"across worker processes). iter_num={args.iter_num} may be slow.")

    if args.eval_all:
        for locname in list(location_dict.keys()):
            gmrc = GMRC(location=locname,
                        iter_num=args.iter_num,
                        particle_altitude=particle_altitude,
                        bfield_type=args.bfield_type,
                        particle_type=plabel,
                        n_workers=args.n_workers,
                        solver=args.solver)
            _run_gmrc(gmrc, args)
    else:
        gmrc = GMRC(location=args.locname,
                    iter_num=args.iter_num,
                    particle_altitude=particle_altitude,
                    bfield_type=args.bfield_type,
                    particle_type=plabel,
                    n_workers=args.n_workers,
                    solver=args.solver)
        _run_gmrc(gmrc, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        'Evaluates the geomagnetic cutoff rigidities of some location for N iterations using a Monte-Carlo sampling scheme, and produces a heatmap for such geomagnetic cutoff rigidities.'
    )
    parser.add_argument('-ln',
                        '--locname',
                        dest="locname",
                        default="Kamioka",
                        type=str,
                        help="Detector location to evaluate GM cutoffs.")
    parser.add_argument('-n',
                        '--iter_num',
                        dest="iter_num",
                        default=50000,
                        type=int,
                        help="Number of iterations for Monte-Carlo.")
    parser.add_argument('-bf',
                        '--bfield',
                        dest="bfield_type",
                        default="igrf",
                        type=str,
                        help="The geomagnetic field model used.")
    parser.add_argument('-a',
                        '--all',
                        dest="eval_all",
                        action="store_true",
                        help="Evaluate GM cutoffs for all locations.")
    parser.add_argument('--show',
                        dest="show_plot",
                        action="store_true",
                        help="Show the plot in an external display.")
    parser.add_argument(
        '-d',
        '--debug',
        dest="debug_mode",
        action="store_true",
        help="Enable debug mode. Sets N = 10 and enable --show=True.")
    parser.add_argument(
        '-w',
        '--workers',
        dest="n_workers",
        default=None,
        type=int,
        help="Number of parallel worker processes (default: physical core count).")
    parser.add_argument(
        '--solver',
        dest="solver",
        default="rk4",
        choices=["rk4", "boris", "rk45"],
        help="Integration method: rk4 (default), boris, rk45 (adaptive).")
    parser.add_argument(
        '--field-mode',
        dest="field_mode",
        default="igrf",
        choices=["igrf", "table"],
        help="Field evaluation mode: igrf (direct IGRF via C++ TrajectoryTracer, default) "
             "or table (precomputed 3-D lookup table via Python RK4; slow, use small -n).")

    args = parser.parse_args()
    eval_gmrc(args)
