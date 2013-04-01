swiglibdir = $(libdir)/swig/openmbvcppinterface/octave

swiglib_LTLIBRARIES = OpenMBV.la

OpenMBV_swig_octave.cc: ../OpenMBV.i
	$(swig) -octave -noglobal -Werror -c++ -I../.. -o OpenMBV_swig_octave.cc ../OpenMBV.i

OpenMBV_la_SOURCES = OpenMBV_swig_octave.cc
OpenMBV_la_LDFLAGS = -module -Wl,-rpath,\$$ORIGIN/../../.. -shrext .oct
OpenMBV_la_CPPFLAGS = $$($(mkoctfile) -p INCFLAGS) $(MBXMLUTILSTINYXML_CFLAGS) -Wno-error=sign-compare -Wno-error=unused-function -Wno-sign-compare -Wno-unused-function
OpenMBV_la_LIBADD = $(OCTAVE_LIBS) ../../openmbvcppinterface/libopenmbvcppinterface.la



TESTS = octavetest.m
