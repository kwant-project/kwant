#! kwant_deps is defined in the top-level meson.build (includes Python & NumPy)

py_files = [
    '__init__.py',
    '_colormaps.py',
    '_common.py',
    '_plotter.py',
    'builder.py',
    'digest.py',
    'kpm.py',
    'lattice.py',
    'plotter.py',
    'qsymm.py',
    'rmt.py',
    'system.py',
    'wraparound.py',
]

py.install_sources(py_files, subdir: 'kwant')

# Cython extensions (Meson handles Cython invocation directly)
foreach module_name : ['_system', 'operator']
    py.extension_module(
        module_name,
        module_name + '.pyx',
        dependencies: kwant_deps,
        include_directories: inc_dirs,
        install: true,
        subdir: 'kwant',
        cython_args: ['-3'],
        # Ensure deprecated NumPy C API usage is disabled at compile time so
        # that NumPy does not emit the "Using deprecated NumPy API" warning.
        # Pass the define to both the C and C++ compilers to be safe.
        c_args: ['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'],
        cpp_args: ['-std=c++14', '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'],
    )
endforeach

subdir('continuum')
subdir('graph')
subdir('linalg')
subdir('physics')
subdir('solvers')
subdir('tests')
