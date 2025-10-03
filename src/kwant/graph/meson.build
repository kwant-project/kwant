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
        c_args: ['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'],
        cpp_args: ['-std=c++14', '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'],
    )
endforeach

subdir('tests')
