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
xmldir = @datadir@/@PACKAGE@/xml
dist_mbxmlutilsschema_DATA = physicalvariable.xsd embed.xsd parameter.xsd
dist_xml_DATA = measurement.xml



# validate measurement.xml
.measurement.xml.isvalid: $(srcdir)/measurement.xml $(srcdir)/measurement.xsd
	rm -f .measurement.xml.isvalid
	$(XC_EXEC_PREFIX) ../mbxmlutilshelper/mbxmlutilsvalidate$(EXEEXT) $$($(XC_PATH_CONVERT) $(abs_srcdir)/measurement.xsd) $$($(XC_PATH_CONVERT) $(abs_srcdir)/measurement.xml) && touch .measurement.xml.isvalid



# generate physicalvariable.xsd
physicalvariable.xsd: $(srcdir)/measurement.xml $(srcdir)/measurement2physicalvariable.xsl .measurement.xml.isvalid
	@XALAN@ -o physicalvariable.xsd $(srcdir)/measurement.xml $(srcdir)/measurement2physicalvariable.xsl



lib_LTLIBRARIES = libmbxmlutils.la libmbxmlutils-eval-octave.la libmbxmlutils-eval-xmlflat.la $(LIBMBXMLUTILS_EVAL_GLOBAL_PYTHON_LA)
module_LTLIBRARIES = fmatvec_symbolic_swig_octave.la registerPath.la
moduledir = $(runlibdir)

# utility library
libmbxmlutils_ladir = $(includedir)/mbxmlutils
libmbxmlutils_la_CPPFLAGS = -I$(top_srcdir) -I.. -I. $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
libmbxmlutils_la_LIBADD = ../mbxmlutilshelper/libmbxmlutilshelper.la $(XERCESC_LIBS) $(FMATVEC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@ $(LIBDL)
libmbxmlutils_la_SOURCES = eval.cc preprocess.cc
libmbxmlutils_la_HEADERS = eval.h preprocess.h

# octave evaluator
libmbxmlutils_eval_octave_ladir = $(includedir)/mbxmlutils
libmbxmlutils_eval_octave_la_CPPFLAGS = -I$(abs_top_srcdir) -I. $(MKOCTFILE_CFLAGS) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS) $(NOWARNING)
libmbxmlutils_eval_octave_la_LIBADD = $(MKOCTFILE_LIBS) libmbxmlutils.la ../mbxmlutilshelper/libmbxmlutilshelper.la $(XERCESC_LIBS) $(FMATVEC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@
libmbxmlutils_eval_octave_la_LDFLAGS = -avoid-version
libmbxmlutils_eval_octave_la_SOURCES = octeval.cc
libmbxmlutils_eval_octave_la_HEADERS = octeval.h
# octave registerPath
registerPath_la_CPPFLAGS = -I$(abs_top_srcdir) -I. $(MKOCTFILE_CFLAGS) $(NOWARNING)
registerPath_la_LIBADD = $(MKOCTFILE_LIBS) libmbxmlutils.la
registerPath_la_LDFLAGS = -module -shrext .oct -avoid-version
registerPath_la_SOURCES = registerPath.cc

# xmlflat dummy evaluator
libmbxmlutils_eval_xmlflat_ladir = $(includedir)/mbxmlutils
libmbxmlutils_eval_xmlflat_la_CPPFLAGS = -I$(top_srcdir) -I. -I.. $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
libmbxmlutils_eval_xmlflat_la_LIBADD = libmbxmlutils.la $(FMATVEC_LIBS) $(XERCESC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@
libmbxmlutils_eval_xmlflat_la_LDFLAGS = -avoid-version
libmbxmlutils_eval_xmlflat_la_SOURCES = xmlflateval.cc
libmbxmlutils_eval_xmlflat_la_HEADERS = xmlflateval.h

# python part of the utility library
libmbxmlutils_eval_global_python_ladir = $(includedir)/mbxmlutils
libmbxmlutils_eval_global_python_la_CPPFLAGS = -I$(top_srcdir) -I. -I.. $(PYTHON_CFLAGS) -I$(PYTHON_LIBDIR)/site-packages/numpy/core/include $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS) -D_hypot=hypot # -D_hypot is a MSVS/python bug workaround
libmbxmlutils_eval_global_python_la_LIBADD = $(PYTHON_LIBS) libmbxmlutils.la ../mbxmlutilshelper/libmbxmlutilshelper.la $(FMATVEC_LIBS) $(XERCESC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@
libmbxmlutils_eval_global_python_la_LDFLAGS = -avoid-version
libmbxmlutils_eval_global_python_la_SOURCES = pyeval.cc
libmbxmlutils_eval_global_python_la_HEADERS = pyeval.h pycppwrapper.h

BUILT_SOURCES = fmatvec_symbolic_swig_octave.cc

# program to convert a complex xml file to a plain xml file
bin_PROGRAMS = mbxmlutilspp
mbxmlutilspp_SOURCES = preprocessor.cc
mbxmlutilspp_CPPFLAGS = -I. -I.. -I$(top_srcdir) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
mbxmlutilspp_LDADD = libmbxmlutils.la ../mbxmlutilshelper/libmbxmlutilshelper.la $(XERCESC_LIBS) -l@BOOST_REGEX_LIB@ -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@ $(FMATVEC_LIBS)



# Newer octave version (>=3.8!?) use #include <config.h> instead of #include "config.h"
# in oct.h. Hence we are not allowed to add -I.. to compile flags since
# this will include config.h from this project (mbxmlutils) instead of the octave
# one. Since -I.. is added as a default include path by automake we must prevent this.
AUTOMAKE_OPTIONS = nostdinc
.DELETE_ON_ERROR: # swig -Werror requires this
fmatvec_symbolic_swig_octave.cc: $(srcdir)/fmatvec_symbolic_swig_octave.i
	rm -f fmatvec_symbolic_swig_octave.cc
	$(swig) -octave -MP -MD -Werror -c++ -globals . -o fmatvec_symbolic_swig_octave.cc $(FMATVEC_CFLAGS) $(srcdir)/fmatvec_symbolic_swig_octave.i
	test @MBXMLUTILS_OCTAVE_MAJOR_VERSION@ -ge 4 -a @MBXMLUTILS_OCTAVE_MINOR_VERSION@ -ge 4 && sed -i -re "s/\<is_object\>/isobject/g" fmatvec_symbolic_swig_octave.cc || exit 0 # WORKAROUND for SWIG bug
-include $(builddir)/fmatvec_symbolic_swig_octave.d

fmatvec_symbolic_swig_octave_la_SOURCES = fmatvec_symbolic_swig_octave.cc
fmatvec_symbolic_swig_octave_la_LDFLAGS = -module -shrext .oct -avoid-version
fmatvec_symbolic_swig_octave_la_CPPFLAGS = $(MKOCTFILE_CFLAGS) $(FMATVEC_CFLAGS) -Wno-deprecated-declarations $(NOWARNING) -DSWIG_OCTAVE_NO_SEGFAULT_HACK -D_USE_MATH_DEFINES
fmatvec_symbolic_swig_octave_la_LIBADD = $(MKOCTFILE_LIBS) $(FMATVEC_LIBS)



CLEANFILES = .measurement.xml.isvalid physicalvariable.xsd fmatvec_symbolic_swig_octave.cc fmatvec_symbolic_swig_octave.d



# generate deplib files for octave oct-files, fmatvec_symbolic_swig_octave, mbxmlutils-eval-octave and registerPath
install-exec-local: install-moduleLTLIBRARIES
	set -e; \
	OCTLIBDIR=$$($(XC_EXEC_PREFIX) $(mkoctfile) -p OCTLIBDIR | dos2unix | tr '\\' '/'); \
        test "$(build)" != "$(host)" && OCTLIBDIR=$$(readlink -f $$(winepath -u $$OCTLIBDIR)); \
	mkdir -p $(runlibdir); \
	for i in $$OCTLIBDIR/oct/*/*.oct; do \
	  test $$(basename $$i) == __init_gnuplot__.oct && continue; \
	  if test $$i -nt $(runlibdir)/$$(basename $$i).deplibs; then \
	    echo "Create dependency file and wrapper file for $$i"; \
	    if [ $(host_os) != "mingw32" ]; then \
	      cp -u $$i $(runlibdir)/$$(basename $$i .oct).so; \
	      $(CC) -shared -fPIC -Wl,--disable-new-dtags,-rpath,\$$ORIGIN/../lib -o $(runlibdir)/$$(basename $$i) \
	        $(runlibdir)/$$(basename $$i .oct).so; \
	    else \
	      cp -u $$i $(runlibdir)/$$(basename $$i); \
	    fi; \
	    python3 $(prefix)/share/mbxmlutils/python/deplibs.py $(runlibdir)/$$(basename $$i) > \
	      $(runlibdir)/$$(basename $$i).deplibs.tmp; \
	    mv -f $(runlibdir)/$$(basename $$i).deplibs.tmp $(runlibdir)/$$(basename $$i).deplibs; \
	  fi; \
	done; \
	cp -u $$OCTLIBDIR/oct/*/PKG_ADD $(runlibdir)
	set -e; \
	for F in $(runlibdir)/fmatvec_symbolic_swig_octave.oct $(runlibdir)/libmbxmlutils-eval-octave$(shext) $(runlibdir)/registerPath.oct $(RUNLIBDIR_LIBMBXMLUTILS_EVAL_GLOBAL_PYTHON_SO); do \
	  if test $$F -nt $$F.deplibs -o ! -e $$F; then \
	    echo "Create dependency file for $$F"; \
	    python3 $(prefix)/share/mbxmlutils/python/deplibs.py $$F > \
	      $$F.deplibs.tmp; \
	    mv -f $$F.deplibs.tmp $$F.deplibs; \
	  fi; \
	done
	set -e; \
	mkdir -p $(prefix)/share/deplibs; \
	for i in $(PYTHON_LIBDIR)/$(PYTHON_PACKAGES)/numpy/*/*.so* \
	         $(PYTHON_LIBDIR)/$(PYTHON_PACKAGES)/numpy/*/*.pyd \
	         $(PYTHON_LIBDIR)/$(PYTHON_PACKAGES)/numpy/*/*.dll; do \
	  if test $$i -nt $(prefix)/share/deplibs/$$(basename $$i).deplibs; then \
	    echo "Create dependency file and wrapper file for $$i"; \
	    python3 $(prefix)/share/mbxmlutils/python/deplibs.py $$i > \
	      $(prefix)/share/deplibs/$$(basename $$i).deplibs.tmp; \
	    mv -f $(prefix)/share/deplibs/$$(basename $$i).deplibs.tmp $(prefix)/share/deplibs/$$(basename $$i).deplibs; \
	  fi; \
	done
