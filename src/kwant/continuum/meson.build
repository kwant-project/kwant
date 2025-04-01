# Install Python modules
py.install_sources([
    '__init__.py',
    'discretizer.py',
    '_common.py',
    'landau_levels.py',
], subdir: 'kwant/continuum')

# Handle tests
tests_exist = run_command('test', '-d', 'tests', check: false).returncode() == 0
if tests_exist
    subdir('tests')
endif
