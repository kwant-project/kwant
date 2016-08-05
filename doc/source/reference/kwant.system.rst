:mod:`kwant.system` -- Low-level interface of systems
*****************************************************

.. automodule:: kwant.system

This module is the binding link between constructing tight-binding systems and
doing calculations with these systems.  It defines the interface which any
problem-solving algorithm should be able to access, independently on how the
system was constructed.  This is achieved by using python abstract base classes
(ABC) -- classes, which help to ensure that any derived classes implement the
necessary interface.

Any system which is provided to a solver should be derived from the appropriate
class in this module, and every solver can assume that its input corresponds to
the interface defined here.

.. autosummary::
   :toctree: generated/

   System
   InfiniteSystem
   FiniteSystem
   PrecalculatedLead
