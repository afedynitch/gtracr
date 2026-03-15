#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "igrf_table.hpp"

struct BatchGMRCParams {
    double latitude, longitude;           // degrees
    double detector_alt, particle_alt;    // km (converted to m internally)
    double escape_radius;                 // meters (default 10*RE)
    double charge;                        // units of e (+1 for proton)
    double mass;                          // GeV/c^2
    double min_rigidity, max_rigidity, delta_rigidity;  // GV
    double dt, max_time;                  // seconds
    char solver_type;                     // 'r', 'b', 'a'
    double atol, rtol;
    int n_samples, n_threads;             // 0 threads = hardware_concurrency()
    int max_attempts_factor;              // safety limit: max attempts = n_samples * this (default 30)
    uint64_t base_seed;
};

struct BatchGMRCResult {
    std::vector<double> zenith, azimuth, rcutoff;  // each size <= n_samples
    int64_t total_trajectories;  // total evaluate() calls across all threads
};

BatchGMRCResult batch_gmrc_evaluate(
    const float* shared_table, const TableParams& table_params,
    const std::pair<std::string, double>& igrf_params,
    const BatchGMRCParams& params);
