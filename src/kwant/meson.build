#! kwant_deps is defined in the top-level meson.build (includes Python & NumPy)

# Compile Python modules - install all Python modules in the current directory
py_files = run_command('find', '.', '-maxdepth', '1', '-name', '*.py', '-not', '-name', '_*system.py', '-not', '-name', 'operator.py', check: true).stdout().strip().split('\n')

# Process the file paths to remove ./ prefix
clean_py_files = []
foreach file : py_files
    if file.strip() != ''
        clean_py_files += [file.strip('./')]
    endif
endforeach

py.install_sources(clean_py_files, subdir: 'kwant')

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
