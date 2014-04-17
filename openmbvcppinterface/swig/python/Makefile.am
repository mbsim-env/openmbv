# Note: wee install the resulting python file in $prefix/bin since this is the only place where Windows searches dependent libraries
swigdir = $(bindir)
swig_LTLIBRARIES = _OpenMBV.la

CLEANFILES=OpenMBV_swig_pyhton.cc OpenMBV.py
OpenMBV_swig_pyhton.cc: ../OpenMBV.i ../../openmbvcppinterface/*.h
	$(swig) -python -Werror -c++ -I../.. -o OpenMBV_swig_pyhton.cc ../OpenMBV.i
OpenMBV.py: OpenMBV_swig_pyhton.cc
	test ! -f $@ && (rm -f OpenMBV_swig_pyhton.cc && $(MAKE) $(AM_MAKEFLAGS) OpenMBV_swig_pyhton.cc) || :

_OpenMBV_la_SOURCES = OpenMBV_swig_pyhton.cc
_OpenMBV_la_LDFLAGS = -module -Wl,-rpath,\$$ORIGIN/../lib -shrext $(PYTHON_SHEXT) -avoid-version
_OpenMBV_la_CPPFLAGS = $(PYTHON_CFLAGS) $(HDF5SERIE_CFLAGS) -I../.. -Wno-error=unused-but-set-variable -Wno-unused-but-set-variable -Wno-error=unused-value -Wno-unused-value
_OpenMBV_la_LIBADD = $(PYTHON_LIBS) ../../openmbvcppinterface/libopenmbvcppinterface.la

install-exec-hook: OpenMBV.py
	$(MKDIR_P) $(swigdir)
	$(INSTALL_DATA) OpenMBV.py $(swigdir)/OpenMBV.py
uninstall-local:
	rm -f $(swigdir)/OpenMBV.py



TESTS = pythontest.sh

exampledir = $(prefix)/share/openmbvcppinterface/examples/swig
example_DATA = pythontest.py
