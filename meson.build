project('kwant', 'c', 'cpp', 'cython',
        version: run_command('python3', '-m', 'setuptools_scm', check: true).stdout().strip(),
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

numpy_dep = dependency('numpy')

# Set include directories - using relative paths for project dirs
inc_dirs = include_directories('src/kwant', 'src/kwant/graph')

# Share dependencies with all submodules (include NumPy)
kwant_deps = [py_dep, numpy_dep]

# Core C/Cython modules
subdir('src')
