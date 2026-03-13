'''
Compare the values of the final times and sixvector of the trajectory for the dipole model
'''

import os
import sys
import numpy as np
import pytest

from gtracr.trajectory import Trajectory

# in the form :
# (plabel, zenith, azimuth, particle_altitude,
# latitude, longitude, detector_altitude, rigidity, kinetic energy)
initial_variable_list = [
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


def test_trajectories_dipole():
    '''
    Test the final times of the trajectory evaluation in the dipole field.
    '''

    expected_times = [
        1e-05, 0.29990000000015915, 0.19221000000005145, 0.20288000000006212,
        0.21143000000007067, 0.2024600000000617, 0.19868000000005792,
        0.2169700000000762, 0.19499000000005423, 0.23231000000009155,
        0.007349999999999869, 0.01941999999999938, 0.19331000000005255
    ]

    dt = 1e-5
    max_time = 1.

    for iexp, initial_variables in enumerate(initial_variable_list):

        (plabel, zenith, azimuth, palt, lat, lng, dalt, rig,
         en) = initial_variables

        traj = Trajectory(plabel=plabel,
                          zenith_angle=zenith,
                          azimuth_angle=azimuth,
                          particle_altitude=palt,
                          latitude=lat,
                          longitude=lng,
                          detector_altitude=dalt,
                          rigidity=rig,
                          energy=en,
                          bfield_type="dipole")

        traj.get_trajectory(dt=dt, max_time=max_time)

        assert np.allclose(traj.final_time, expected_times[iexp])


def test_trajectories_igrf():
    '''
    Test the final times of the trajectory evaluation in the IGRF field.
    '''

    expected_times = [
        1e-05, 0.29990000000015915, 0.19221000000005145, 0.20288000000006212,
        0.21143000000007067, 0.2024600000000617, 0.19868000000005792,
        0.2169700000000762, 0.19499000000005423, 0.23231000000009155,
        0.007349999999999869, 0.01941999999999938, 0.19331000000005255
    ]

    dt = 1e-5
    max_time = 1.

    for iexp, initial_variables in enumerate(initial_variable_list):

        (plabel, zenith, azimuth, palt, lat, lng, dalt, rig,
         en) = initial_variables

        traj = Trajectory(plabel=plabel,
                          zenith_angle=zenith,
                          azimuth_angle=azimuth,
                          particle_altitude=palt,
                          latitude=lat,
                          longitude=lng,
                          detector_altitude=dalt,
                          rigidity=rig,
                          energy=en,
                          bfield_type="igrf")

        traj.get_trajectory(dt=dt, max_time=max_time)

        assert np.allclose(traj.final_time, expected_times[iexp])


def test_trajectories_stepsize():
    '''
    Test the final times of the trajectory evaluation in the igrf field for
    different step sizes
    '''

    expected_times = [
        0.22073792992447885, 0.22073800000531751, 0.22073800000020008,
        0.22074000000007998, 0.220799999999992, 0.22100000000000017,
        0.23000000000000007, 0.30000000000000004
    ]

    dt_arr = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    max_time = 1.

    (plabel, zenith, azimuth, palt, lat, lng, dalt, rig,
     en) = ("p+", 90., 0., 100., 0., 0., 0., 50., None)

    for iexp, dt in enumerate(dt_arr):

        traj = Trajectory(plabel=plabel,
                          zenith_angle=zenith,
                          azimuth_angle=azimuth,
                          particle_altitude=palt,
                          latitude=lat,
                          longitude=lng,
                          detector_altitude=dalt,
                          rigidity=rig,
                          energy=en,
                          bfield_type="igrf")

        traj.get_trajectory(dt=dt, max_time=max_time)

        assert np.allclose(traj.final_time, expected_times[iexp])


def test_trajectories_maxtimes():
    '''
    Test the final times of the trajectory evaluation in the igrf field for
    different maximal times
    '''

    expected_times = [
        0.00999999999999976, 0.027829999999999036, 0.07743000000000268,
        0.2154500000000747, 0.22074000000007998, 0.22074000000007998,
        0.22074000000007998, 0.22074000000007998, 0.22074000000007998,
        0.22074000000007998
    ]

    dt = 1e-5
    max_times = np.logspace(-2, 2, 10)

    for iexp, max_time in enumerate(max_times):

        (plabel, zenith, azimuth, palt, lat, lng, dalt, rig,
         en) = ("p+", 90., 0., 100., 0., 0., 0., 50., None)

        traj = Trajectory(plabel=plabel,
                          zenith_angle=zenith,
                          azimuth_angle=azimuth,
                          particle_altitude=palt,
                          latitude=lat,
                          longitude=lng,
                          detector_altitude=dalt,
                          rigidity=rig,
                          energy=en,
                          bfield_type="igrf")

        traj.get_trajectory(dt=dt, max_time=max_time)

        assert np.allclose(traj.final_time, expected_times[iexp])



def test_trajectories_dates():
    '''
    Test the final times of the trajectory evaluation in the igrf field for
    different dates
    '''

    expected_times = [
        0.22074000000007998, 0.22074000000007998, 0.22074000000007998,
        0.22074000000007998, 0.22074000000007998, 0.22074000000007998,
        0.22074000000007998, 0.22074000000007998, 0.22074000000007998,
        0.22074000000007998
    ]

    dt = 1e-5
    max_time = 1.

    dates = [
        "1900-01-01", "1909-01-01", "1900-10-31", "2020-09-12", "2004-03-08",
        "2000-02-28", "1970-03-26", "1952-04-31", "1999-03-08", "2024-03-09"
    ]
    for iexp, date in enumerate(dates):

        (plabel, zenith, azimuth, palt, lat, lng, dalt, rig,
         en) = ("p+", 90., 0., 100., 0., 0., 0., 50., None)

        traj = Trajectory(plabel=plabel,
                          zenith_angle=zenith,
                          azimuth_angle=azimuth,
                          particle_altitude=palt,
                          latitude=lat,
                          longitude=lng,
                          detector_altitude=dalt,
                          rigidity=rig,
                          energy=en,
                          bfield_type="igrf",
                          date=date)

        traj.get_trajectory(dt=dt, max_time=max_time)

        assert np.allclose(traj.final_time, expected_times[iexp])


def test_dipole_sixvec():

    expected_sixvec = [
        [6.47119963e+06, 1.57079633e+00, 4.63047878e-04, -6.02521771e-21, 9.86721733e-34, 1.60799992e-17],
        [6.37140834e+07, 1.57079633e+00, 5.40386360e+00, 1.55538056e-17, 1.78905251e-33, 4.08174548e-18],
        [6.37128147e+07, 7.97913976e-01, 2.13090889e+00, 2.67480631e-17, 7.52597807e-19, 1.49064674e-18],
        [6.37121036e+07, 2.58654734e-01, -1.90446521e+00, 1.07180485e-17, -3.55880769e-19, -6.66658090e-20],
        [6.37124285e+07, 4.61065831e+00, 6.28478521e+10, 3.58811687e-06, 3.66274018e-07, 1.74067724e-11],
        [6.37134067e+07, 1.61295116e+00, -2.79464187e+00, 1.06205640e-17, 5.72463040e-19, -1.36299661e-18],
        [6.37131288e+07, 9.04985289e-01, -2.18987050e+00, 1.06097158e-17, 8.87738404e-19, 1.26817872e-18],
        [6.37137998e+07, 1.73157584e+00, -1.61993612e+00, 1.05135464e-17, 1.50396793e-18, 1.48513171e-18],
        [6.37137381e+07, 4.01854827e-01, -1.52694936e+00, 1.07185670e-17, 1.87736741e-19, -1.15019830e-19],
        [6.37120552e+07, 1.55291035e+00, 4.89727764e+00, 1.03367764e-17, -1.02235437e-18, 2.65536315e-18],
        [6.46893596e+06, 1.73738224e+00, 3.33505808e+00, -2.45605250e-18, 7.25645703e-19, 8.01689444e-19],
        [6.46903298e+06, 1.75544342e+00, 3.69383927e+00, -4.84443873e-18, -2.01797763e-18, 9.89615683e-19],
        [6.37147793e+07, 9.08048550e-01, -1.24126416e+00, 2.67178408e-17, 1.18891577e-18, 1.65404679e-18],
    ]

    dt = 1e-5
    max_time = 1.

    for iexp, initial_variables in enumerate(initial_variable_list):

        (plabel, zenith, azimuth, palt, lat, lng, dalt, rig, en) = initial_variables

        traj = Trajectory(
            plabel=plabel,
            zenith_angle=zenith,
            azimuth_angle=azimuth,
            particle_altitude=palt,
            latitude=lat,
            longitude=lng,
            detector_altitude=dalt,
            rigidity=rig,
            energy=en,
            bfield_type="dipole",
        )

        traj.get_trajectory(dt=dt, max_time=max_time)

        assert np.allclose(traj.final_sixvector, np.array(expected_sixvec[iexp]), rtol=1e-5)
