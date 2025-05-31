# Newer octave version (>=3.8!?) use #include <config.h> instead of #include "config.h"
# in oct.h. Hence we are not allowed to add -I$(abs_top_srcdir) to compile flags since
# this will include config.h from this project (openmbvcppinterface) instead of the octave
# one. Since top_srcdir is added as a default include path by automake we must prevent this.
AUTOMAKE_OPTIONS = nostdinc

# Note: wee install the resulting *.oct file in $prefix/bin since this is the only place where Windows searches dependent libraries
swigdir = $(bindir)
swig_LTLIBRARIES = OpenMBV.la

BUILT_SOURCES = OpenMBV_swig_octave.cc

CLEANFILES = OpenMBV_swig_octave.cc OpenMBV_swig_octave.d
.DELETE_ON_ERROR: # swig -Werror requires this
OpenMBV_swig_octave.cc: $(srcdir)/../OpenMBV.i
	rm -f OpenMBV_swig_octave.cc
	$(swig) -octave -MP -MD -Werror -c++ -I$(top_srcdir) $(FMATVEC_CFLAGS) -o OpenMBV_swig_octave.cc $(srcdir)/../OpenMBV.i
-include $(builddir)/OpenMBV_swig_octave.d

OpenMBV_la_SOURCES = OpenMBV_swig_octave.cc
OpenMBV_la_LDFLAGS = -module -shrext .oct -avoid-version
OpenMBV_la_CPPFLAGS = -I$(top_srcdir) $(OCTAVE_FLAGS) $(HDF5SERIE_CFLAGS) $(MBXMLUTILSHELPER_CFLAGS) -Wno-error=sign-compare -Wno-error=unused-function -Wno-sign-compare -Wno-unused-function -Wno-error=unused-value -Wno-unused-value -Wno-error=deprecated-declarations $(NOWARNING) -DSWIG_OCTAVE_NO_SEGFAULT_HACK -D_USE_MATH_DEFINES
OpenMBV_la_LIBADD = $(LDFLAGS_NOUNDEF) $(OCTAVE_LIBS) $(HDF5SERIE_LIBS) $(MBXMLUTILSHELPER_LIBS) ../../openmbvcppinterface/libopenmbvcppinterface.la



TESTS = octavetest.sh

exampledir = $(prefix)/share/openmbvcppinterface/examples/swig
example_DATA = $(srcdir)/octavetest.m
