SUBDIRS = octave

if MBXMLUTILS_MINGW
  runlibdir = $(bindir)
else
  runlibdir = $(libdir)
endif

# distribute and install
schemadir = @datadir@/@PACKAGE@/schema/http___openmbv_berlios_de_MBXMLUtils_physicalvariable
xmldir = @datadir@/@PACKAGE@/xml
swigrefdir = $(runlibdir)/@swig_ref
dist_schema_DATA = physicalvariable.xsd embed.xsd parameter.xsd casadi.xsd
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
	../mbxmlutilshelper/mbxmlutilsvalidate$(EXEEXT) $(srcdir)/measurement.xsd $(srcdir)/measurement.xml && touch .measurement.xml.isvalid



# generate physicalvariable.xsd
physicalvariable.xsd: $(srcdir)/measurement.xml $(srcdir)/measurement2physicalvariable.xsl .measurement.xml.isvalid
	@XALAN@ -o physicalvariable.xsd $(srcdir)/measurement.xml $(srcdir)/measurement2physicalvariable.xsl



lib_LTLIBRARIES = libmbxmlutils.la casadi_oct.la

# utility library
libmbxmlutils_ladir = $(includedir)/mbxmlutils
libmbxmlutils_la_CPPFLAGS = -I$(top_srcdir) -I.. -I. $(CASADI_CFLAGS) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
libmbxmlutils_la_LIBADD = ../mbxmlutilshelper/libmbxmlutilshelper.la libmbxmlutils-octave.la $(CASADI_LIBS) $(XERCESC_LIBS) $(FMATVEC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@
libmbxmlutils_la_SOURCES = eval.cc preprocess.cc
libmbxmlutils_la_HEADERS = eval.h preprocess.h

# octave part of the utility library
libmbxmlutils_octave_ladir = $(includedir)/mbxmlutils
noinst_LTLIBRARIES = libmbxmlutils-octave.la
libmbxmlutils_octave_la_CPPFLAGS = -I$(top_srcdir) -I. $(MKOCTFILE_CFLAGS) $(CASADI_CFLAGS) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
libmbxmlutils_octave_la_LIBADD = $(MKOCTFILE_LIBS)
libmbxmlutils_octave_la_SOURCES = octeval.cc
libmbxmlutils_octave_la_HEADERS = octeval.h

# program to convert a complex xml file to a plain xml file
bin_PROGRAMS = mbxmlutilspp
mbxmlutilspp_SOURCES = preprocessor.cc
mbxmlutilspp_CPPFLAGS = -I$(top_srcdir) $(MKOCTFILE_CFLAGS) $(CASADI_CFLAGS) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
mbxmlutilspp_LDADD = libmbxmlutils.la ../mbxmlutilshelper/libmbxmlutilshelper.la $(CASADI_LIBS) $(MKOCTFILE_LIBS) $(XERCESC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@



# Newer octave version (>=3.8!?) use #include <config.h> instead of #include "config.h"
# in oct.h. Hence we are not allowed to add -I.. to compile flags since
# this will include config.h from this project (mbxmlutils) instead of the octave
# one. Since -I.. is added as a default include path by automake we must prevent this.
AUTOMAKE_OPTIONS = nostdinc
casadi_oct_swig_octave.cc: $(srcdir)/casadi_oct.i
	$(swig) -octave -Werror -c++ -o casadi_oct_swig_octave.cc -DCASADI_EXPORT $(CASADI_CFLAGS) $(srcdir)/casadi_oct.i || (rm -f casadi_oct_swig_octave.cc; exit 1)

casadi_oct_la_SOURCES = casadi_oct_swig_octave.cc
casadi_oct_la_LDFLAGS = -module -shrext .oct -avoid-version
casadi_oct_la_CPPFLAGS = $(MKOCTFILE_CFLAGS) $(CASADI_CFLAGS) -Wno-unused-value -Wno-error=maybe-uninitialized -Wno-unused-function
casadi_oct_la_LIBADD = $(MKOCTFILE_LIBS) $(CASADI_LIBS)



CLEANFILES = .measurement.xml.isvalid physicalvariable.xsd casadi_oct_swig_octave.cc



# generate deplib files for octave oct-files and casaid.oct
install-exec-local:
	set -e; \
	OCTLIBDIR=$$($(mkoctfile) -p OCTLIBDIR | dos2unix | tr '\\' '/'); \
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
	if test $(runlibdir)/casadi_oct.oct -nt $(runlibdir)/casadi_oct.oct.deplibs; then \
	  echo "Create dependency file for $(runlibdir)/casadi_oct.oct"; \
	  python $(prefix)/share/mbxmlutils/python/deplibs.py $(runlibdir)/casadi_oct.oct > \
	    $(runlibdir)/casadi_oct.oct.deplibs.tmp; \
	  mv -f $(runlibdir)/casadi_oct.oct.deplibs.tmp $(runlibdir)/casadi_oct.oct.deplibs; \
	fi
