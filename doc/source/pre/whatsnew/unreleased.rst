What's new in next version of Kwant
===================================

This article explains the user-visible changes after Kwant 1.4.4.

Update of the setup script and version requirements
---------------------------------------------------

Following the recommendation of py.test, the command ``setup.py test`` is now
removed. Instead the users should run ``py.test`` directly or use
``import kwant; kwant.test()``.

The minimum required version of Python is now 3.8 (the previous versions are
past end of life).  The other packages required by Kwant are also updated to
their versions available for Python 3.8.

Removal of Umfpack support
--------------------------
Scipy used to provide a Python interface to the Umfpack library.  This is now
done by a separate package ``scikit-umfpack``.  Because it is hard for end
users to obtain this, we have removed built-in support for Umfpack.