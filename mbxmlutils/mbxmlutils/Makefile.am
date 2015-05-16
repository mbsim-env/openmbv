SUBDIRS = octave

if MBXMLUTILS_MINGW
  runlibdir = $(bindir)
else
  runlibdir = $(libdir)
endif

# distribute and install
schemadir = @datadir@/@PACKAGE@/schema/http___openmbv_berlios_de_MBXMLUtils_physicalvariable
xmldir = @datadir@/@PACKAGE@/xml
swigrefdir = $(runlibdir)
dist_schema_DATA = physicalvariable.xsd embed.xsd parameter.xsd casadi.xsd
dist_xml_DATA = measurement.xml
dist_swigref_DATA = @swig_ref





# validate measurement.xml
.measurement.xml.isvalid: $(srcdir)/measurement.xml $(srcdir)/measurement.xsd
	rm -f .measurement.xml.isvalid
	../mbxmlutilshelper/mbxmlutilsvalidate$(EXEEXT) $(srcdir)/measurement.xsd $(srcdir)/measurement.xml && touch .measurement.xml.isvalid



# generate physicalvariable.xsd
physicalvariable.xsd: $(srcdir)/measurement.xml $(srcdir)/measurement2physicalvariable.xsl .measurement.xml.isvalid
	@XALAN@ -o physicalvariable.xsd $(srcdir)/measurement.xml $(srcdir)/measurement2physicalvariable.xsl



# utility library
lib_LTLIBRARIES = libmbxmlutils.la casadi.la
libmbxmlutils_ladir = $(includedir)/mbxmlutils
libmbxmlutils_la_CPPFLAGS = -I$(top_srcdir) $(MKOCTFILE_CFLAGS) $(CASADI_CFLAGS) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
libmbxmlutils_la_LIBADD = ../mbxmlutilshelper/libmbxmlutilshelper.la $(CASADI_LIBS) $(MKOCTFILE_LIBS) $(XERCESC_LIBS) $(FMATVEC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@
libmbxmlutils_la_SOURCES = octeval.cc preprocess.cc
libmbxmlutils_la_HEADERS = octeval.h preprocess.h

# program to convert a complex xml file to a plain xml file
bin_PROGRAMS = mbxmlutilspp
mbxmlutilspp_SOURCES = preprocessor.cc
mbxmlutilspp_CPPFLAGS = -I$(top_srcdir) $(MKOCTFILE_CFLAGS) $(CASADI_CFLAGS) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
mbxmlutilspp_LDADD = libmbxmlutils.la ../mbxmlutilshelper/libmbxmlutilshelper.la $(CASADI_LIBS) $(MKOCTFILE_LIBS) $(XERCESC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@



# Newer octave version (>=3.8!?) use #include <config.h> instead of #include "config.h"
# in oct.h. Hence we are not allowed to add -I$(abs_top_srcdir) to compile flags since
# this will include config.h from this project (openmbvcppinterface) instead of the octave
# one. Since top_srcdir is added as a default include path by automake we must prevent this.
AUTOMAKE_OPTIONS = nostdinc
casadi_swig_octave.cc: $(srcdir)/casadi.i
	$(swig) -octave -Werror -c++ -o casadi_swig_octave.cc -DCASADI_EXPORT $(CASADI_CFLAGS) $(srcdir)/casadi.i || (rm -f casadi_swig_octave.cc; exit 1)

casadi_la_SOURCES = casadi_swig_octave.cc
casadi_la_LDFLAGS = -module -shrext .oct -avoid-version
casadi_la_CPPFLAGS = $(MKOCTFILE_CFLAGS) $(CASADI_CFLAGS)
casadi_la_LIBADD = $(MKOCTFILE_LIBS) $(CASADI_LIBS)



CLEANFILES = .measurement.xml.isvalid physicalvariable.xsd casadi_swig_octave.cc



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
	if test $(runlibdir)/casadi.oct -nt $(runlibdir)/casadi.oct.deplibs; then \
	  echo "Create dependency file for $(runlibdir)/casadi.oct"; \
	  python $(prefix)/share/mbxmlutils/python/deplibs.py $(runlibdir)/casadi.oct > \
	    $(runlibdir)/casadi.oct.deplibs.tmp; \
	  mv -f $(runlibdir)/casadi.oct.deplibs.tmp $(runlibdir)/casadi.oct.deplibs; \
	fi
