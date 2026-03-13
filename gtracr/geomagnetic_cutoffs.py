import sys
import os
import math
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm
from datetime import date
from concurrent.futures import ProcessPoolExecutor, as_completed

from gtracr.trajectory import Trajectory
from gtracr.lib._libgtracr import TrajectoryTracer
from gtracr.lib.constants import (
    ELEMENTARY_CHARGE,
    KG_PER_GEVC2,
    EARTH_RADIUS,
)
from gtracr.utils import particle_dict, ymd_to_dec

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

# ---------------------------------------------------------------------------
# Process-level TrajectoryTracer cache
# ---------------------------------------------------------------------------
# Each worker process stores one shared TrajectoryTracer here.  It is
# initialised once by _init_worker() (via ProcessPoolExecutor's initializer
# argument) so the expensive IGRF JSON load happens only *once per worker
# process* rather than once per trajectory.  The tracer is then reset and its
# start_altitude updated before every evaluate() call.
_worker_tracer = None
_worker_max_step = None


def _init_worker(common_params):
    """Initialise a single TrajectoryTracer for the lifetime of this worker.

    Called automatically by ProcessPoolExecutor when a new worker process
    starts.  Storing the tracer in a module-level variable avoids the
    IGRF JSON parse (the dominant construction cost) for every trajectory.

    Parameters
    ----------
    common_params : tuple
        ``(plabel, bfield_type, date_str, palt, dt, max_time)``
        All values that are identical across every MC sample in a GMRC run.
    """
    global _worker_tracer, _worker_max_step

    plabel, bfield_type, date_str, palt, dt, max_time = common_params

    particle = particle_dict[plabel]
    # Use an arbitrary rigidity just to get charge/mass (they don't change).
    particle.set_from_rigidity(10.0)
    charge_si = particle.charge * ELEMENTARY_CHARGE
    mass_si = particle.mass * KG_PER_GEVC2

    # Nominal start_alt; updated per direction via set_start_altitude().
    nominal_start_alt = palt * 1e3
    esc_alt = 10.0 * EARTH_RADIUS
    max_step = math.ceil(max_time / dt)
    bfield_char = bfield_type[0]

    datapath = os.path.abspath(os.path.join(CURRENT_DIR, "data"))
    dec_date = ymd_to_dec(date_str)
    igrf_params = (datapath, dec_date)

    _worker_tracer = TrajectoryTracer(
        charge_si,
        mass_si,
        nominal_start_alt,
        esc_alt,
        dt,
        max_step,
        bfield_char,
        igrf_params,
    )
    _worker_max_step = max_step


def _evaluate_single_direction(args):
    """Evaluate the cutoff rigidity for a single random (zenith, azimuth)
    direction, reusing a cached TrajectoryTracer to avoid rebuilding the
    IGRF model for every trajectory.

    Two levels of caching are employed:

    1. **Process-level cache** (medium-complexity optimisation): when a worker
       was initialised by ``_init_worker`` the module-level ``_worker_tracer``
       is shared across *all* MC samples handled by that worker.  Only
       ``set_start_altitude()`` and ``reset()`` are called between samples.

    2. **Per-direction cache** (easy optimisation / sequential fallback): when
       no process-level tracer exists (e.g. ``n_workers=1`` sequential path),
       a single TrajectoryTracer is constructed for the current direction and
       reused across the rigidity scan.  This alone reduces IGRF constructions
       from ``n_rigidities`` per direction to 1.

    Parameters
    ----------
    args : tuple
        ``(location, plabel, bfield_type, date_str, palt, rigidity_list,
        dt, max_time, seed)``

    Returns
    -------
    (azimuth, zenith, rcutoff) or (azimuth, zenith, 0.0) if no cutoff found
    """
    location, plabel, bfield_type, date_str, palt, rigidity_list, dt, max_time, seed = args
    rng = np.random.default_rng(seed)
    azimuth, zenith = rng.random(2) * np.array([360.0, 180.0])

    global _worker_tracer

    # Build a reference Trajectory for the first rigidity to obtain
    # direction-specific parameters (start_alt, charge_si, mass_si, …).
    # The start_alt depends only on zenith angle, so it is the same for every
    # rigidity within this direction.
    _ref_traj = Trajectory(
        plabel=plabel,
        location_name=location,
        zenith_angle=zenith,
        azimuth_angle=azimuth,
        particle_altitude=palt,
        rigidity=rigidity_list[0],
        bfield_type=bfield_type,
        date=date_str,
    )

    if _worker_tracer is not None:
        # --- Process-level cache: update direction-specific start_altitude ---
        tracer = _worker_tracer
        tracer.set_start_altitude(_ref_traj.start_alt)
    else:
        # --- Per-direction cache: create one tracer for this direction ---
        charge_si = _ref_traj.charge * ELEMENTARY_CHARGE
        mass_si = _ref_traj.mass * KG_PER_GEVC2
        max_step = math.ceil(max_time / dt)
        tracer = TrajectoryTracer(
            charge_si,
            mass_si,
            _ref_traj.start_alt,
            _ref_traj.esc_alt,
            dt,
            max_step,
            _ref_traj.bfield_type,
            _ref_traj.igrf_params,
        )

    # Evaluate the first rigidity using the reference trajectory's six-vector.
    tracer.reset()
    tracer.evaluate(0.0, _ref_traj.particle_sixvector)
    if tracer.particle_escaped:
        return (azimuth, zenith, rigidity_list[0])

    # Scan remaining rigidities, reusing the same tracer.
    for rigidity in rigidity_list[1:]:
        traj = Trajectory(
            plabel=plabel,
            location_name=location,
            zenith_angle=zenith,
            azimuth_angle=azimuth,
            particle_altitude=palt,
            rigidity=rigidity,
            bfield_type=bfield_type,
            date=date_str,
        )
        tracer.reset()
        tracer.evaluate(0.0, traj.particle_sixvector)
        if tracer.particle_escaped:
            return (azimuth, zenith, rigidity)

    return (azimuth, zenith, 0.0)


class GMRC():
    '''
    Evaluates the geomagnetic cutoff rigidities associated to a specific location on the globe for each zenith and azimuthal angle (a zenith angle > 90 degrees are for upward-moving particles, that is, for cosmic rays coming from the other side of Earth).

    The cutoff rigidities are evaluated using a Monte-Carlo sampling scheme, combined with a 2-dimensional linear interpolation using `scipy.interpolate`.

    The resulting cutoffs can be plotted as 2-dimensional heatmap.

    Parameters
    -----------

    - location : str
        The location in which the geomagnetic cutoff rigidities are evaluated (default = "Kamioka"). The names must be one of the locations contained in `location_dict`, which is configured in `gtracr.utils`.
    - particle_altitude : float
        The altitude in which the cosmic ray interacts with the atmosphere in km (default = 100).
    - iter_num : int
        The number of iterations to perform for the Monte-Carlo sampling routine (default = 10000)
    - bfield_type : str
        The type of magnetic field model to use for the evaluation of the cutoff rigidities (default = "igrf"). Set to "dipole" to use the dipole approximation of the geomagnetic field instead.
    - particle_type : str
        The type of particle of the cosmic ray (default  ="p+").
    - date : str
        The specific date in which the geomagnetic rigidity cutoffs are evaluated. Defaults to the current date.
    - min_rigidity : float
        The minimum rigidity to which we evaluate the cutoff rigidities for (default = 5 GV).
    - max_rigidity : float
        The maximum rigidity to which we evaluate the cutoff rigidities for (default = 55 GV).
    - delta_rigidity : float
        The spacing between each rigidity (default = 5 GV). Sets the coarseness of the rigidity sample space.
    - n_workers : int, optional
        Number of parallel worker processes to use. Defaults to the number of CPU cores.
        Set to 1 to disable parallelism (useful for debugging).
    '''
    def __init__(self,
                 location="Kamioka",
                 particle_altitude=100,
                 iter_num=10000,
                 bfield_type="igrf",
                 particle_type="p+",
                 date=str(date.today()),
                 min_rigidity=5.,
                 max_rigidity=55.,
                 delta_rigidity=1.,
                 n_workers=None):
        # set class attributes
        self.location = location
        self.palt = particle_altitude
        self.iter_num = iter_num
        self.bfield_type = bfield_type
        self.plabel = particle_type
        self.date = date
        self.n_workers = n_workers  # None = use all CPU cores
        '''
        Rigidity configurations
        '''
        self.rmin = min_rigidity
        self.rmax = max_rigidity
        self.rdelta = delta_rigidity

        # generate list of rigidities
        self.rigidity_list = np.arange(self.rmin, self.rmax, self.rdelta)

        # initialize container for rigidity cutoffs
        self.data_dict = {
            "azimuth": np.zeros(self.iter_num),
            "zenith": np.zeros(self.iter_num),
            "rcutoff": np.zeros(self.iter_num)
        }

    def evaluate(self, dt=1e-5, max_time=1):
        '''
        Evaluate the rigidity cutoff value at some provided location
        on Earth for a given cosmic ray particle.

        Uses parallel worker processes (one per CPU core by default) to
        evaluate independent Monte Carlo samples concurrently.  A single
        TrajectoryTracer is created per worker process via an initializer,
        avoiding repeated IGRF JSON loading across trajectories.

        Parameters
        ----------

        - dt : float
            The stepsize of each trajectory evaluation (default = 1e-5)
        - max_time : float
            The maximal time of each trajectory evaluation (default = 1.).

        '''
        rigidity_list = list(self.rigidity_list)

        # Build argument list for each MC sample, using distinct random seeds
        # for reproducibility across different worker processes
        rng = np.random.default_rng()
        seeds = rng.integers(0, 2**31, size=self.iter_num)

        args_list = [
            (self.location, self.plabel, self.bfield_type, self.date,
             self.palt, rigidity_list, dt, max_time, int(seeds[i]))
            for i in range(self.iter_num)
        ]

        # Parameters shared by every MC sample; passed to the worker
        # initializer so each process builds its TrajectoryTracer only once.
        common_params = (
            self.plabel,
            self.bfield_type,
            self.date,
            self.palt,
            dt,
            max_time,
        )

        n_workers = self.n_workers
        use_parallel = (n_workers is None or n_workers > 1)

        if use_parallel:
            # Parallel evaluation: each worker process evaluates one MC sample.
            # The initializer creates one cached TrajectoryTracer per worker,
            # eliminating repeated IGRF construction inside each task.
            results = []
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(common_params,),
            ) as executor:
                future_to_idx = {
                    executor.submit(_evaluate_single_direction, args): i
                    for i, args in enumerate(args_list)
                }
                for future in tqdm(as_completed(future_to_idx),
                                   total=self.iter_num,
                                   desc="GMRC evaluation"):
                    i = future_to_idx[future]
                    results.append((i, future.result()))

            for i, (az, zen, rc) in results:
                self.data_dict["azimuth"][i] = az
                self.data_dict["zenith"][i] = zen
                self.data_dict["rcutoff"][i] = rc
        else:
            # Sequential fallback (n_workers=1), useful for debugging.
            # _evaluate_single_direction still creates one TrajectoryTracer per
            # direction (per-direction cache), which is far cheaper than one
            # per rigidity.
            for i in tqdm(range(self.iter_num)):
                az, zen, rc = _evaluate_single_direction(args_list[i])
                self.data_dict["azimuth"][i] = az
                self.data_dict["zenith"][i] = zen
                self.data_dict["rcutoff"][i] = rc

    def interpolate_results(self,
                            method="linear",
                            ngrid_azimuth=70,
                            ngrid_zenith=70):
        '''
        Interpolate the rigidity cutoffs using `scipy.interpolate.griddata`

        Parameters
        ----------
        - method : str
            The type of linear interpolation used for `griddata` (default = "linear"). Choices are between "nearest", "linear", and "cubic".
        - ngrid_azimuth, ngrid_zenith : int
            The number of grids for the azimuth and zenith angles used for the interpolation (default = 70).

        Returns
        --------

        Returns a tuple of the following objects:

        - azimuth_grid : np.array(float), size ngrid_azimuth
            The linearly spaced values of the azimuthal angle
        - zenith_grid : np.array(float), size ngrid_zenith
            The linearly spaced values of the zenith angle
        - rcutoff_grid : np.array(float), size ngrid_azimuth x ngrid_zenith
            The interpolated geomagnetic cutoff rigidities.
        '''

        azimuth_grid = np.linspace(np.min(self.data_dict["azimuth"]),
                                   np.max(self.data_dict["azimuth"]),
                                   ngrid_azimuth)
        zenith_grid = np.linspace(np.max(self.data_dict["zenith"]),
                                  np.min(self.data_dict["zenith"]),
                                  ngrid_zenith)

        rcutoff_grid = griddata(points=(self.data_dict["azimuth"],
                                        self.data_dict["zenith"]),
                                values=self.data_dict["rcutoff"],
                                xi=(azimuth_grid[None, :], zenith_grid[:,
                                                                       None]),
                                method=method)

        return (azimuth_grid, zenith_grid, rcutoff_grid)
