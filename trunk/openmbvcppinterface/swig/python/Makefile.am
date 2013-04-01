swiglibdir = $(libdir)/swig/openmbvcppinterface/python

swiglib_LTLIBRARIES = _OpenMBV.la

OpenMBV_swig_pyhton.cc: ../OpenMBV.i
	$(swig) -python -Werror -c++ -I../.. -o OpenMBV_swig_pyhton.cc ../OpenMBV.i

_OpenMBV_la_SOURCES = OpenMBV_swig_pyhton.cc
_OpenMBV_la_LDFLAGS = -module
_OpenMBV_la_CPPFLAGS = $(PYTHON_CFLAGS) $(MBXMLUTILSTINYXML_CFLAGS) -Wno-error=unused-but-set-variable
_OpenMBV_la_LIBADD = $(PYTHON_LIBS) ../../openmbvcppinterface/libopenmbvcppinterface.la

install-exec-local:
	$(INSTALL_DATA) OpenMBV.py $(swiglibdir)
uninstall-local:
	rm -f $(swiglibdir)/OpenMBV.py



TESTS = pythontest.py
