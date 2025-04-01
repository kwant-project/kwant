# Define kwant dependencies
kwant_deps = [py_dep, numpy_dep]

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

# Cython extensions
cython_modules = [
    ['_system.pyx', 'c++'],
    ['operator.pyx', 'c++']
]

foreach mod : cython_modules
    src_file = mod[0]
    lang = mod[1]
    module_name = src_file.split('.')[0]

    if lang == 'c++'
        output_ext = '.cpp'
        args = ['-3', '--cplus']
    else
        output_ext = '.c'
        args = ['-3']
    endif

    cython_gen = custom_target(
        module_name + '_cython',
        input: src_file,
        output: module_name + output_ext,
        command: [cython] + args + ['@INPUT@', '-o', '@OUTPUT@'],
    )

    py.extension_module(
        module_name,
        cython_gen,
        dependencies: kwant_deps,
        include_directories: inc_dirs,
        install: true,
        subdir: 'kwant',
        cpp_args: lang == 'c++' ? ['-std=c++14'] : []
    )
endforeach

# Process subdirectories - use run_command to check if directories exist
continuum_exists = run_command('test', '-d', 'continuum', check: false).returncode() == 0
graph_exists = run_command('test', '-d', 'graph', check: false).returncode() == 0
linalg_exists = run_command('test', '-d', 'linalg', check: false).returncode() == 0
physics_exists = run_command('test', '-d', 'physics', check: false).returncode() == 0
solvers_exists = run_command('test', '-d', 'solvers', check: false).returncode() == 0
tests_exist = run_command('test', '-d', 'tests', check: false).returncode() == 0


if continuum_exists
    subdir('continuum')
endif

if graph_exists
    subdir('graph')
endif

if linalg_exists
    subdir('linalg')
endif

if physics_exists
    subdir('physics')
endif

if solvers_exists
    subdir('solvers')
endif

if tests_exist
    subdir('tests')
endif
