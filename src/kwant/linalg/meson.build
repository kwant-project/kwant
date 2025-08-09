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
    cpp_args: ['-std=c++14'] + numpy_args,
)

subdir('tests')
