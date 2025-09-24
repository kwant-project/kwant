# Install Python modules
py.install_sources([
    '__init__.py',
    'defs.py',
], subdir: 'kwant/graph')

# Compile Cython extensions
foreach module_name : ['core', 'dijkstra']
    py.extension_module(
        module_name,
        module_name + '.pyx',
        dependencies: kwant_deps,
        include_directories: inc_dirs,
        install: true,
        subdir: 'kwant/graph',
        cython_args: ['-3'],
        cpp_args: ['-std=c++14'],
    )
endforeach

subdir('tests')
