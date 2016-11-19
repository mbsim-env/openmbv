# Note: wee install the resulting python file in $prefix/bin since this is the only place where Windows searches dependent libraries
swigdir = $(bindir)
swig_LTLIBRARIES = _OpenMBV.la

CLEANFILES=OpenMBV_swig_python.cc OpenMBV.py OpenMBV_swig_python.d
OpenMBV_swig_python.cc: $(srcdir)/../OpenMBV.i $(top_srcdir)/openmbvcppinterface/*.h
	rm -f OpenMBV_swig_python.cc
	$(swig) -python -MP -MD -Werror -c++ -I$(top_srcdir) $(FMATVEC_CFLAGS) -o OpenMBV_swig_python.cc $(srcdir)/../OpenMBV.i
-include OpenMBV_swig_python.d
OpenMBV.py: OpenMBV_swig_python.cc
	test ! -f $@ && (rm -f OpenMBV_swig_python.cc && $(MAKE) $(AM_MAKEFLAGS) OpenMBV_swig_python.cc) || :

_OpenMBV_la_SOURCES = OpenMBV_swig_python.cc
_OpenMBV_la_LDFLAGS = -module -shrext $(PYTHON_SHEXT) -avoid-version
_OpenMBV_la_CPPFLAGS = $(PYTHON_CFLAGS) $(HDF5SERIE_CFLAGS) $(MBXMLUTILSHELPER_CFLAGS) -I$(top_srcdir) -Wno-error=unused-but-set-variable -Wno-unused-but-set-variable -Wno-error=unused-value -Wno-unused-value -D_hypot=hypot # -D_hypot is a MSVS/python bug workaround
_OpenMBV_la_LIBADD = $(PYTHON_LIBS) ../../openmbvcppinterface/libopenmbvcppinterface.la

install-exec-hook: OpenMBV.py
	$(MKDIR_P) $(swigdir)
	$(INSTALL_DATA) OpenMBV.py $(swigdir)/OpenMBV.py
uninstall-local:
	rm -f $(swigdir)/OpenMBV.py



TESTS = pythontest.sh

exampledir = $(prefix)/share/openmbvcppinterface/examples/swig
example_DATA = $(srcdir)/pythontest.py
