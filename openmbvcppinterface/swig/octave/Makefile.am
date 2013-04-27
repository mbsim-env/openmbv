# Note: wee install the resulting *.oct file in $prefix/bin since this is the only place where Windows searches dependent libraries
swigdir = $(bindir)
swig_LTLIBRARIES = OpenMBV.la

CLEANFILES=OpenMBV_swig_octave.cc
OpenMBV_swig_octave.cc: ../OpenMBV.i
	$(swig) -octave -Werror -c++ -I../.. -o OpenMBV_swig_octave.cc ../OpenMBV.i

OpenMBV_la_SOURCES = OpenMBV_swig_octave.cc
OpenMBV_la_LDFLAGS = -module -Wl,-rpath,\$$ORIGIN/../lib -shrext .oct -avoid-version
OpenMBV_la_CPPFLAGS = $(OCTAVE_FLAGS) $(MBXMLUTILSTINYXML_CFLAGS) $(HDF5SERIE_CFLAGS) -I../.. -Wno-error=sign-compare -Wno-error=unused-function -Wno-sign-compare -Wno-unused-function -Wno-error=unused-value -Wno-unused-value
OpenMBV_la_LIBADD = $(OCTAVE_LIBS) ../../openmbvcppinterface/libopenmbvcppinterface.la



TESTS = octavetest.sh

exampledir = $(prefix)/share/openmbvcppinterface/examples/swig
example_DATA = octavetest.m
