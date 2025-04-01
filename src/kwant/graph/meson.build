# Install Python modules
py.install_sources([
    '__init__.py',
    'defs.py',
], subdir: 'kwant/graph')

# Compile Cython extensions
cython_modules = [
    ['core.pyx', 'c++'],
    ['dijkstra.pyx', 'c++']
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
        'graph_' + module_name + '_cython',
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
        subdir: 'kwant/graph',
        cpp_args: lang == 'c++' ? ['-std=c++14'] : []
    )
endforeach

# Handle tests
tests_exist = run_command('test', '-d', 'tests', check: false).returncode() == 0
if tests_exist
    subdir('tests')
endif
