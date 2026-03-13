py.extension_module(
  '_libgtracr',
  sources: [
    'src/TrajectoryTracer.cpp',
    'src/igrf.cpp',
    'src/pybind11_wrapper.cpp',
  ],
  include_directories: [
    include_directories('include'),
    include_directories('extern'),   # for nlohmann/json.hpp
  ],
  dependencies: [py.dependency(), pybind11_dep],
  cpp_args: extra_cpp_args,
  install: true,
  subdir: 'gtracr/lib',
)

py.install_sources(
  [
    '__init__.py',
    'constants.py',
    'igrf_utils.py',
    'location.py',
    'magnetic_field.py',
    'particle.py',
    'trajectory_tracer.py',
  ],
  subdir: 'gtracr/lib',
)
