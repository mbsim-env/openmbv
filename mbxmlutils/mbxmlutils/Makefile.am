if COND_WIN32
  runlibdir = $(bindir)
  shext = .dll
else
  runlibdir = $(libdir)
  shext = .so
endif

if COND_PYTHON
  PYTHON = python
  LIBMBXMLUTILS_EVAL_GLOBAL_PYTHON_LA = libmbxmlutils-eval-global-python.la
  RUNLIBDIR_LIBMBXMLUTILS_EVAL_GLOBAL_PYTHON_SO = $(runlibdir)/libmbxmlutils-eval-global-python$(shext)
endif

SUBDIRS = octave $(PYTHON)

# distribute and install
mbxmlutilsschemadir = @datadir@/@PACKAGE@/schema/http___www_mbsim-env_de_MBXMLUtils
casadischemadir = @datadir@/@PACKAGE@/schema/http___www_mbsim-env_de_MBXMLUtils_CasADi
xmldir = @datadir@/@PACKAGE@/xml
swigrefdir = $(runlibdir)/@swig_ref
dist_mbxmlutilsschema_DATA = physicalvariable.xsd embed.xsd parameter.xsd
dist_casadischema_DATA = casadi.xsd
dist_xml_DATA = measurement.xml
dist_swigref_DATA = \
  @swig_ref/abs.m @swig_ref/acosh.m @swig_ref/acos.m @swig_ref/asinh.m @swig_ref/asin.m @swig_ref/atan2.m \
  @swig_ref/atanh.m @swig_ref/atan.m @swig_ref/ceil.m @swig_ref/cosh.m @swig_ref/cos.m @swig_ref/erf.m \
  @swig_ref/exp.m @swig_ref/floor.m @swig_ref/fmod.m @swig_ref/horzcat.m @swig_ref/log10.m @swig_ref/log.m \
  @swig_ref/max.m @swig_ref/min.m @swig_ref/power.m @swig_ref/sign.m @swig_ref/sinh.m @swig_ref/sin.m \
  @swig_ref/sqrt.m @swig_ref/tanh.m @swig_ref/tan.m @swig_ref/vertcat.m





# validate measurement.xml
.measurement.xml.isvalid: $(srcdir)/measurement.xml $(srcdir)/measurement.xsd
	rm -f .measurement.xml.isvalid
	$(XC_EXEC_PREFIX) ../mbxmlutilshelper/mbxmlutilsvalidate$(EXEEXT) $(srcdir)/measurement.xsd $(srcdir)/measurement.xml && touch .measurement.xml.isvalid



# generate physicalvariable.xsd
physicalvariable.xsd: $(srcdir)/measurement.xml $(srcdir)/measurement2physicalvariable.xsl .measurement.xml.isvalid
	@XALAN@ -o physicalvariable.xsd $(srcdir)/measurement.xml $(srcdir)/measurement2physicalvariable.xsl



lib_LTLIBRARIES = libmbxmlutils.la libmbxmlutils-eval-octave.la libmbxmlutils-eval-xmlflat.la $(LIBMBXMLUTILS_EVAL_GLOBAL_PYTHON_LA)
module_LTLIBRARIES = casadi_oct.la registerPath.la
moduledir = $(runlibdir)

# utility library
libmbxmlutils_ladir = $(includedir)/mbxmlutils
libmbxmlutils_la_CPPFLAGS = -I$(top_srcdir) -I.. -I. $(CASADI_CFLAGS) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
libmbxmlutils_la_LIBADD = ../mbxmlutilshelper/libmbxmlutilshelper.la $(CASADI_LIBS) $(XERCESC_LIBS) $(FMATVEC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@
libmbxmlutils_la_SOURCES = eval.cc preprocess.cc
libmbxmlutils_la_HEADERS = eval.h preprocess.h

# octave evaluator
libmbxmlutils_eval_octave_ladir = $(includedir)/mbxmlutils
libmbxmlutils_eval_octave_la_CPPFLAGS = -I$(abs_top_srcdir) -I. $(MKOCTFILE_CFLAGS) $(CASADI_CFLAGS) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
libmbxmlutils_eval_octave_la_LIBADD = $(MKOCTFILE_LIBS) libmbxmlutils.la ../mbxmlutilshelper/libmbxmlutilshelper.la $(CASADI_LIBS) $(XERCESC_LIBS) $(FMATVEC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@
libmbxmlutils_eval_octave_la_LDFLAGS = -avoid-version
libmbxmlutils_eval_octave_la_SOURCES = octeval.cc
libmbxmlutils_eval_octave_la_HEADERS = octeval.h
# octave registerPath
registerPath_la_CPPFLAGS = -I$(abs_top_srcdir) -I. $(MKOCTFILE_CFLAGS)
registerPath_la_LIBADD = $(MKOCTFILE_LIBS) libmbxmlutils.la
registerPath_la_LDFLAGS = -module -shrext .oct -avoid-version
registerPath_la_SOURCES = registerPath.cc

# xmlflat dummy evaluator
libmbxmlutils_eval_xmlflat_ladir = $(includedir)/mbxmlutils
libmbxmlutils_eval_xmlflat_la_CPPFLAGS = -I$(top_srcdir) -I. -I.. $(CASADI_CFLAGS) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
libmbxmlutils_eval_xmlflat_la_LIBADD = libmbxmlutils.la $(FMATVEC_LIBS) $(XERCESC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@
libmbxmlutils_eval_xmlflat_la_LDFLAGS = -avoid-version
libmbxmlutils_eval_xmlflat_la_SOURCES = xmlflateval.cc
libmbxmlutils_eval_xmlflat_la_HEADERS = xmlflateval.h

# python part of the utility library
libmbxmlutils_eval_global_python_ladir = $(includedir)/mbxmlutils
libmbxmlutils_eval_global_python_la_CPPFLAGS = -I$(top_srcdir) -I. -I.. $(PYTHON_CFLAGS) -I$(PYTHON_LIBDIR)/site-packages/numpy/core/include $(CASADI_CFLAGS) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS) -D_hypot=hypot # -D_hypot is a MSVS/python bug workaround
libmbxmlutils_eval_global_python_la_LIBADD = $(PYTHON_LIBS) libmbxmlutils.la ../mbxmlutilshelper/libmbxmlutilshelper.la $(FMATVEC_LIBS) $(XERCESC_LIBS) $(CASADI_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@
libmbxmlutils_eval_global_python_la_LDFLAGS = -avoid-version
libmbxmlutils_eval_global_python_la_SOURCES = pyeval.cc
libmbxmlutils_eval_global_python_la_HEADERS = pyeval.h py2py3cppwrapper.h

BUILT_SOURCES = casadi_oct_swig_octave.cc swigpyrun.h

swigpyrun.h:
	$(swig) -python -external-runtime swigpyrun.h

# program to convert a complex xml file to a plain xml file
bin_PROGRAMS = mbxmlutilspp
mbxmlutilspp_SOURCES = preprocessor.cc
mbxmlutilspp_CPPFLAGS = -I. -I.. -I$(top_srcdir) $(CASADI_CFLAGS) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
mbxmlutilspp_LDADD = libmbxmlutils.la ../mbxmlutilshelper/libmbxmlutilshelper.la $(CASADI_LIBS) $(XERCESC_LIBS) -l@BOOST_REGEX_LIB@ -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@ $(FMATVEC_LIBS)



# Newer octave version (>=3.8!?) use #include <config.h> instead of #include "config.h"
# in oct.h. Hence we are not allowed to add -I.. to compile flags since
# this will include config.h from this project (mbxmlutils) instead of the octave
# one. Since -I.. is added as a default include path by automake we must prevent this.
AUTOMAKE_OPTIONS = nostdinc
casadi_oct_swig_octave.cc: $(srcdir)/casadi_oct.i
	rm -f casadi_oct_swig_octave.cc
	$(swig) -octave -MP -MD -Werror -c++ -o casadi_oct_swig_octave.cc -DCASADI_EXPORT $(CASADI_CFLAGS) $(srcdir)/casadi_oct.i
-include $(builddir)/casadi_oct_swig_octave.d

casadi_oct_la_SOURCES = casadi_oct_swig_octave.cc
casadi_oct_la_LDFLAGS = -module -shrext .oct -avoid-version
casadi_oct_la_CPPFLAGS = $(MKOCTFILE_CFLAGS) $(CASADI_CFLAGS) -Wno-unused-value -Wno-error=maybe-uninitialized -Wno-unused-function -Wno-error=deprecated-declarations
casadi_oct_la_LIBADD = $(MKOCTFILE_LIBS) $(CASADI_LIBS)



CLEANFILES = .measurement.xml.isvalid physicalvariable.xsd casadi_oct_swig_octave.cc casadi_oct_swig_octave.d swigpyrun.h



# generate deplib files for octave oct-files, casadi_oct, mbxmlutils-eval-octave and registerPath
install-exec-local: install-moduleLTLIBRARIES
	set -e; \
	OCTLIBDIR=$$($(XC_EXEC_PREFIX) $(mkoctfile) -p OCTLIBDIR | dos2unix | tr '\\' '/'); \
        test "$(build)" != "$(host)" && OCTLIBDIR=$$(readlink -f $$(winepath -u $$OCTLIBDIR)); \
	mkdir -p $(runlibdir); \
	for i in $$OCTLIBDIR/oct/*/*.oct; do \
	  if test $$i -nt $(runlibdir)/$$(basename $$i).deplibs; then \
	    echo "Create dependency file and wrapper file for $$i"; \
	    if [ $(host_os) != "mingw32" ]; then \
	      cp -u $$i $(runlibdir)/$$(basename $$i .oct).so; \
	      $(CC) -shared -fPIC -Wl,--disable-new-dtags,-rpath,\$$ORIGIN/../lib -o $(runlibdir)/$$(basename $$i) \
	        $(runlibdir)/$$(basename $$i .oct).so; \
	    else \
	      cp -u $$i $(runlibdir)/$$(basename $$i); \
	    fi; \
	    python $(prefix)/share/mbxmlutils/python/deplibs.py $(runlibdir)/$$(basename $$i) > \
	      $(runlibdir)/$$(basename $$i).deplibs.tmp; \
	    mv -f $(runlibdir)/$$(basename $$i).deplibs.tmp $(runlibdir)/$$(basename $$i).deplibs; \
	  fi; \
	done; \
	cp -u $$OCTLIBDIR/oct/*/PKG_ADD $(runlibdir)
	set -e; \
	for F in $(runlibdir)/casadi_oct.oct $(runlibdir)/libmbxmlutils-eval-octave$(shext) $(runlibdir)/registerPath.oct $(RUNLIBDIR_LIBMBXMLUTILS_EVAL_GLOBAL_PYTHON_SO); do \
	  if test $$F -nt $$F.deplibs -o ! -e $$F; then \
	    echo "Create dependency file for $$F"; \
	    python $(prefix)/share/mbxmlutils/python/deplibs.py $$F > \
	      $$F.deplibs.tmp; \
	    mv -f $$F.deplibs.tmp $$F.deplibs; \
	  fi; \
	done
