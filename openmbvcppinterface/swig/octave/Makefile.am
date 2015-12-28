# Newer octave version (>=3.8!?) use #include <config.h> instead of #include "config.h"
# in oct.h. Hence we are not allowed to add -I$(abs_top_srcdir) to compile flags since
# this will include config.h from this project (openmbvcppinterface) instead of the octave
# one. Since top_srcdir is added as a default include path by automake we must prevent this.
AUTOMAKE_OPTIONS = nostdinc

# Note: wee install the resulting *.oct file in $prefix/bin since this is the only place where Windows searches dependent libraries
swigdir = $(bindir)
swig_LTLIBRARIES = OpenMBV.la

CLEANFILES=OpenMBV_swig_octave.cc OpenMBV_swig_octave.d
OpenMBV_swig_octave.cc: $(srcdir)/../OpenMBV.i
	$(swig) -octave -MD -Werror -c++ $(FMATVEC_CFLAGS) -I$(top_srcdir) -o OpenMBV_swig_octave.cc $(srcdir)/../OpenMBV.i || (rm -f OpenMBV_swig_octave.cc; exit 1)
-include OpenMBV_swig_octave.d

OpenMBV_la_SOURCES = OpenMBV_swig_octave.cc
OpenMBV_la_LDFLAGS = -module -shrext .oct -avoid-version
OpenMBV_la_CPPFLAGS = $(OCTAVE_FLAGS) $(HDF5SERIE_CFLAGS) $(MBXMLUTILSHELPER_CFLAGS) -I$(top_srcdir) -Wno-error=sign-compare -Wno-error=unused-function -Wno-sign-compare -Wno-unused-function -Wno-error=unused-value -Wno-unused-value
OpenMBV_la_LIBADD = $(OCTAVE_LIBS) ../../openmbvcppinterface/libopenmbvcppinterface.la



TESTS = octavetest.sh

exampledir = $(prefix)/share/openmbvcppinterface/examples/swig
example_DATA = octavetest.m
