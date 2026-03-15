# igrf_table.cpp is CPU-only and always compiled; GTRACR_HAS_GPU is defined
# only when hipcc is present (future: batch_kernel.hip goes there).
gpu_cpp_args = []
if has_gpu
  gpu_cpp_args = ['-DGTRACR_HAS_GPU']
endif

py.extension_module(
  '_libgtracr',
  sources: [
    'src/TrajectoryTracer.cpp',
    'src/BatchGMRC.cpp',
    'src/igrf.cpp',
    'src/pybind11_wrapper.cpp',
    'gpu/igrf_table.cpp',
  ],
  include_directories: [
    include_directories('include'),
    include_directories('extern'),   # for nlohmann/json.hpp
    include_directories('gpu'),      # for igrf_table.hpp
  ],
  dependencies: [py.dependency(), pybind11_dep],
  cpp_args: extra_cpp_args + gpu_cpp_args,
  install: true,
  subdir: 'gtracr/lib',
)

py.install_sources(
  [
    '__init__.py',
    'constants.py',
    'igrf_table.py',
    'igrf_utils.py',
    'location.py',
    'magnetic_field.py',
    'particle.py',
    'trajectory_tracer.py',
  ],
  subdir: 'gtracr/lib',
)
