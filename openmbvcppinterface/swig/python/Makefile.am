swiglibdir = $(libdir)/swig/openmbvcppinterface/python

swiglib_LTLIBRARIES = _OpenMBV.la

CLEANFILES=OpenMBV_swig_pyhton.cc OpenMBV.py
OpenMBV_swig_pyhton.cc: ../OpenMBV.i
	$(swig) -python -Werror -c++ -I../.. -o OpenMBV_swig_pyhton.cc ../OpenMBV.i
OpenMBV.py: OpenMBV_swig_pyhton.cc
	test ! -f $@ && (rm -f OpenMBV_swig_pyhton.cc && $(MAKE) $(AM_MAKEFLAGS) OpenMBV_swig_pyhton.cc) || :

_OpenMBV_la_SOURCES = OpenMBV_swig_pyhton.cc
_OpenMBV_la_LDFLAGS = -module -Wl,-rpath,\$$ORIGIN/../../.. -shrext $(PYTHONSHEXT) -avoid-version
_OpenMBV_la_CPPFLAGS = $(PYTHON_CFLAGS) $(MBXMLUTILSTINYXML_CFLAGS) $(HDF5SERIE_CFLAGS) -I../.. -Wno-error=unused-but-set-variable -Wno-unused-but-set-variable -Wno-error=unused-value -Wno-unused-value
_OpenMBV_la_LIBADD = $(PYTHON_LIBS) ../../openmbvcppinterface/libopenmbvcppinterface.la

install-exec-hook: OpenMBV.py
	$(MKDIR_P) $(swiglibdir)
	$(INSTALL_DATA) OpenMBV.py $(swiglibdir)/OpenMBV.py
uninstall-local:
	rm -f $(swiglibdir)/OpenMBV.py



TESTS = pythontest.sh
