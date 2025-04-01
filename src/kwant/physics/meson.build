# Install Python modules
py.install_sources([
    '__init__.py',
    'noise.py',
    'leads.py',
    'dispersion.py',
    'symmetry.py',
    'gauge.py',
], subdir: 'kwant/physics')

# Handle tests
tests_exist = run_command('test', '-d', 'tests', check: false).returncode() == 0
if tests_exist
    subdir('tests')
endif
