# Install Python modules
py.install_sources([
    '__init__.py',
    'decomp_schur.py',
    'fortran_helpers.py',
    'lll.py',
], subdir: 'kwant/linalg')

# Cython extension for lapack
lapack_cython = custom_target(
    'lapack_cython',
    input: 'lapack.pyx',
    output: 'lapack.cpp',
    command: [cython, '-3', '--cplus', '@INPUT@', '-o', '@OUTPUT@'],
)

py.extension_module(
    'lapack',
    lapack_cython,
    dependencies: kwant_deps,
    include_directories: inc_dirs,
    install: true,
    subdir: 'kwant/linalg',
    cpp_args: ['-std=c++14']
)

# Handle tests
tests_exist = run_command('test', '-d', 'tests', check: false).returncode() == 0
if tests_exist
    subdir('tests')
endif
