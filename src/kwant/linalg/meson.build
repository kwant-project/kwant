# Install Python modules
py.install_sources([
    '__init__.py',
    'decomp_schur.py',
    'fortran_helpers.py',
    'lll.py',
], subdir: 'kwant/linalg')

# Cython extension for lapack
py.extension_module(
    'lapack',
    'lapack.pyx',
    dependencies: kwant_deps,
    include_directories: inc_dirs,
    install: true,
    subdir: 'kwant/linalg',
    cython_args: ['-3'],
    c_args: ['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'],
    cpp_args: ['-std=c++14', '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'],
)

subdir('tests')
