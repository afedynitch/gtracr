'''
Numerical regression tests with tight tolerances.
Golden values were captured from the frozen-field RK4 implementation (2026-03).
These tests verify that optimizations do not change physical results.
'''

import math
import numpy as np
import pytest

from gtracr.lib._libgtracr import IGRF
from gtracr.trajectory import Trajectory

DATA_PATH = 'gtracr/data/igrf13.json'

# ---------------------------------------------------------------------------
# B-field component tests
# ---------------------------------------------------------------------------

# (r [m], theta [rad], phi [rad])
BFIELD_COORDS = [
    (6.471e6, 1.570796326794897, 0.000000000000000),
    (6.471e6, 0.785398163397448, 1.570796326794897),
    (6.471e6, 2.356194490192345, 3.141592653589793),
    (6.471e6, 0.523598775598299, -1.047197551196598),
    (6.471e6, 2.617993877991494, 0.523598775598299),
    (7.000e6, 1.570796326794897, 0.785398163397448),
    (8.000e6, 1.047197551196598, -0.523598775598299),
    (6.471e6, 0.100000000000000, 0.500000000000000),
    (6.471e6, 3.041592653589793, 1.500000000000000),
    (6.600e6, 1.570796326794897, 3.141592653589793),
]

# (Br [T], Btheta [T], Bphi [T])
EXPECTED_BFIELD = [
    (1.4855695312886571e-05, -2.6275042651584387e-05, -2.1783012782847714e-06),
    (-5.022698287666843e-05, -2.2136022738173563e-05, 7.670865558988388e-07),
    (4.998055401773263e-05, -1.7598037567259555e-05, 8.735265501917444e-06),
    (-5.159088313804225e-05, -1.0660607633772066e-05, -4.726147369878846e-06),
    (2.9451320469060627e-05, -1.23928974362497e-05, -1.116463946819446e-05),
    (7.098511625973886e-06, -2.428599076702391e-05, -7.740370143861933e-07),
    (-1.3781812451778208e-05, -1.3714018905197889e-05, -2.416042461267865e-06),
    (-5.3446224075424234e-05, -3.7948639669908606e-06, 1.5139925094950205e-06),
    (5.042536466649683e-05, 4.981942197366722e-06, -1.4294509205006488e-05),
    (3.086411652001845e-06, -3.0081399087441946e-05, 5.111542735065936e-06),
]


@pytest.mark.parametrize("idx", range(len(BFIELD_COORDS)))
def test_igrf_bfield_components(idx):
    r, theta, phi = BFIELD_COORDS[idx]
    br_exp, btheta_exp, bphi_exp = EXPECTED_BFIELD[idx]
    igrf = IGRF(DATA_PATH, 2020.0)
    br, btheta, bphi = igrf.values(r, theta, phi)
    assert np.isclose(br,     br_exp,     rtol=1e-10), f"Br mismatch at coord {idx}"
    assert np.isclose(btheta, btheta_exp, rtol=1e-10), f"Btheta mismatch at coord {idx}"
    assert np.isclose(bphi,   bphi_exp,   rtol=1e-10), f"Bphi mismatch at coord {idx}"


# ---------------------------------------------------------------------------
# IGRF trajectory sixvector tests
# ---------------------------------------------------------------------------

# (plabel, zenith, azimuth, palt, lat, lng, dalt, rig, energy)
INITIAL_VARIABLES = [
    ("p+", 90., 90., 100., 0., 0., 0., 30., None),
    ("p+", 120., 90., 100., 0., 0., -1., 30., None),
    ("p+", 0., 25., 100., 50., 100., 0., 50., None),
    ("p+", 90., 5., 100., 89., 20., 0., 20., None),
    ("p+", 90., 5., 100., -90., 20., 0., 20., None),
    ("e-", 90., 5., 100., 40., 200., 0., 20., None),
    ("p+", 45., 265., 0., 40., 200., 0., 20., None),
    ("p+", 45., 180., 10., 40., 200., 0., 20., None),
    ("p+", 45., 0., 0., 89., 0., 0., 20., None),
    ("p+", 45., 0., 0., 0., 180., 100., 20., None),
    ("p+", 45., 0., 0., 0., 180., 100., 5., None),
    ("p+", 45., 0., 0., 0., 180., 100., None, 10.),
    ("p+", 9., 80., 0., 50., 260., 100., None, 50.),
]

EXPECTED_IGRF_SIXVEC = [
    [6471199.68050328, 1.5707962577201013, 0.0004630478707908207, -5.143554660183891e-21, -7.196105447122918e-21, 1.6079998378099462e-17],
    [6395317.776380827, 1.3992345968129658, 1.981536736603477, -8.57617028687763e-18, -3.0192755183307807e-18, 1.3264636783553269e-17],
    [63712212.04716667, 0.8074398754110363, 2.1770706661221237, 2.6731275728192177e-17, 8.661331197341019e-19, 1.7138007247769652e-18],
    [63713745.74427799, 0.20724228888363305, -1.995680145880181, 1.0713377296526772e-17, -4.1091361933258483e-19, -2.477684921690336e-19],
    [63712428.50813812, 4.610658305768503, 62847852133.84116, 3.58811687217861e-06, 3.6627401823020585e-07, 1.74067739971675e-11],
    [63712062.14375874, 1.4614010872104917, -2.8564244609530656, 1.062865281436195e-17, 2.785079633559938e-19, -1.3921413578559089e-18],
    [63713162.071241274, 1.0062315032130158, -2.2985756659462218, 1.0607067126380323e-17, 1.0148562839204593e-18, 1.1916180072907381e-18],
    [63714514.42363828, 1.789976579331972, -1.7889839111347252, 1.0520224828087366e-17, 1.6731271616622827e-18, 1.2365190297782258e-18],
    [63713052.77406037, 0.3535713883919112, -1.5677032739819432, 1.0717718954147296e-17, 9.910528979051715e-20, -2.3790156902143063e-19],
    [63714469.36343867, 1.3926107403194006, 5.802187144909359, 1.027697986092815e-17, -1.949313944752653e-19, 3.0505145092851058e-18],
    [6471153.899520735, 1.7062073271222196, 3.2713493690618245, -2.3571523815195386e-18, 7.356255264195253e-19, 1.0525277131133863e-18],
    [6470843.307605463, 1.7885568479545975, 3.5009965210797325, -4.656275497747911e-18, -1.0617199839204792e-18, 2.391155319604603e-18],
    [63714794.90253662, 0.898022507357986, -1.319147823449617, 2.6738635877979995e-17, 1.0610687942570718e-18, 1.3822830178447632e-18],
]

EXPECTED_IGRF_ESCAPED = [False, False, True, True, True, True, True, True, True, True, False, False, True]


@pytest.mark.parametrize("idx", range(len(INITIAL_VARIABLES)))
def test_igrf_sixvector(idx):
    plabel, zenith, azimuth, palt, lat, lng, dalt, rig, en = INITIAL_VARIABLES[idx]
    traj = Trajectory(
        plabel=plabel, zenith_angle=zenith, azimuth_angle=azimuth,
        particle_altitude=palt, latitude=lat, longitude=lng,
        detector_altitude=dalt, rigidity=rig, energy=en,
        bfield_type="igrf",
    )
    traj.get_trajectory(dt=1e-5, max_time=1.)
    assert np.allclose(traj.final_sixvector, EXPECTED_IGRF_SIXVEC[idx], rtol=1e-10), \
        f"sixvector mismatch at case {idx}"


@pytest.mark.parametrize("idx", range(len(INITIAL_VARIABLES)))
def test_igrf_escaped_flag(idx):
    plabel, zenith, azimuth, palt, lat, lng, dalt, rig, en = INITIAL_VARIABLES[idx]
    traj = Trajectory(
        plabel=plabel, zenith_angle=zenith, azimuth_angle=azimuth,
        particle_altitude=palt, latitude=lat, longitude=lng,
        detector_altitude=dalt, rigidity=rig, energy=en,
        bfield_type="igrf",
    )
    traj.get_trajectory(dt=1e-5, max_time=1.)
    assert traj.particle_escaped == EXPECTED_IGRF_ESCAPED[idx], \
        f"escaped flag mismatch at case {idx}: got {traj.particle_escaped}, expected {EXPECTED_IGRF_ESCAPED[idx]}"
