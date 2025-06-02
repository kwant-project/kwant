project('kwant', 'c', 'cpp', 'cython',
        version: run_command('python', '-m', 'setuptools_scm', check: true).stdout().strip(),
        license: 'BSD-2-Clause',
        default_options: [
            'warning_level=2',
            'c_std=c11',
            'cpp_std=c++14',
        ],
)

# Find the Python installation
py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

# Get NumPy include directory directly from Python
# This is more compatible with older NumPy versions
# See: https://github.com/mesonbuild/meson/issues/9598
# TODO: Make numpy an actual dependency once we depend on numpy >= 2.0
numpy_inc = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

# Find Cython
cython = find_program('cython')

# Set include directories - using relative paths for project dirs
inc_dirs = include_directories('.', 'src', 'src/kwant', 'src/kwant/graph')

# Define NumPy compiler arguments
numpy_args = ['-I' + numpy_inc]

# Share dependencies with all submodules
kwant_deps = [py_dep]

# Store the NumPy include directory to pass to extensions
numpy_include_dir = numpy_inc

# Core C/Cython modules
subdir('src')
