swiglibdir = $(libdir)/swig/openmbvcppinterface/python

swiglib_LTLIBRARIES = _OpenMBV.la

OpenMBV_swig_pyhton.cc: ../OpenMBV.i
	$(swig) -python -Werror -c++ -I../.. -o OpenMBV_swig_pyhton.cc ../OpenMBV.i

_OpenMBV_la_SOURCES = OpenMBV_swig_pyhton.cc
_OpenMBV_la_LDFLAGS = -module
_OpenMBV_la_CPPFLAGS = $(PYTHON_CFLAGS) $(MBXMLUTILSTINYXML_CFLAGS) -Wno-error=unused-but-set-variable -Wno-unused-but-set-variable -Wno-error=unused-value -Wno-unused-value
_OpenMBV_la_LIBADD = $(PYTHON_LIBS) ../../openmbvcppinterface/libopenmbvcppinterface.la

install-exec-hook:
	$(MKDIR_P) $(swiglibdir)
	$(INSTALL_DATA) OpenMBV.py $(swiglibdir)/OpenMBV.py
uninstall-local:
	rm -f $(swiglibdir)/OpenMBV.py



TESTS = pythontest.py
