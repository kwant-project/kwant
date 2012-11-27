__all__ = ['solve', 'ldos', 'wave_func']

# MUMPS usually works best.  Use SciPy as fallback.
try:
    from . import mumps as smodule
except ImportError:
    from . import sparse as smodule

hidden_instance = smodule.Solver()

solve = hidden_instance.solve
ldos = hidden_instance.ldos
wave_func = hidden_instance.wave_func
