project('gtracr', 'cpp',
  version: '1.0.0',
  default_options: [
    'cpp_std=c++11',
    'optimization=3',
    'warning_level=0',
  ]
)

py = import('python').find_installation(pure: false)
pybind11_dep = dependency('pybind11', fallback: ['pybind11', 'pybind11_dep'])

# Replicate platform-specific compiler flags from setup.py
cc = meson.get_compiler('cpp')
extra_cpp_args = []
if host_machine.system() != 'windows'
  foreach flag : [
    '-march=native', '-ffast-math', '-funroll-loops',
    '-Wno-shorten-64-to-32', '-Wno-parentheses',
    '-Wno-unused-variable', '-Wno-sign-compare',
    '-Wno-cpp', '-Wno-deprecated-declarations',
  ]
    if cc.has_argument(flag)
      extra_cpp_args += flag
    endif
  endforeach
  if host_machine.system() == 'darwin'
    extra_cpp_args += '-stdlib=libc++'
  endif
endif

subdir('gtracr')
