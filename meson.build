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

# NumPy: use Meson's dependency object so include dirs & compile args are added automatically
# Older code tried to manually construct include args via a numpy_inc variable; that's brittle.
numpy_dep = dependency('numpy')

# Obtain NumPy include directory explicitly (Meson does not always add it to compiler args)
numpy_inc = run_command(py, '-c', 'import numpy, sys; sys.stdout.write(numpy.get_include())', check: true).stdout().strip()

# Compiler arguments to add NumPy includes
numpy_args = ['-I' + numpy_inc]

# Find Cython
cython = find_program('cython')

# Set include directories - using relative paths for project dirs
inc_dirs = include_directories('.', 'src', 'src/kwant', 'src/kwant/graph')

# Share dependencies with all submodules (include NumPy)
kwant_deps = [py_dep, numpy_dep]

# Core C/Cython modules
subdir('src')
