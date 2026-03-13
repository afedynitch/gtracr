py.install_sources(
  [
    '__init__.py',
    'geomagnetic_cutoffs.py',
    'plotting.py',
    'trajectory.py',
    'utils.py',
    'version.py',
  ],
  subdir: 'gtracr',
)

install_subdir('data', install_dir: py.get_install_dir() / 'gtracr')

subdir('lib')
subdir('scripts')
subdir('tests')
