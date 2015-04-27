SUBDIRS = octave

# distribute and install
schemadir = @datadir@/@PACKAGE@/schema/http___openmbv_berlios_de_MBXMLUtils_physicalvariable
xmldir = @datadir@/@PACKAGE@/xml
dist_schema_DATA = physicalvariable.xsd embed.xsd parameter.xsd casadi.xsd
dist_xml_DATA = measurement.xml





# validate measurement.xml
.measurement.xml.isvalid: $(srcdir)/measurement.xml $(srcdir)/measurement.xsd
	rm -f .measurement.xml.isvalid
	../mbxmlutilshelper/mbxmlutilsvalidate$(EXEEXT) $(srcdir)/measurement.xsd $(srcdir)/measurement.xml && touch .measurement.xml.isvalid



# generate physicalvariable.xsd
physicalvariable.xsd: $(srcdir)/measurement.xml $(srcdir)/measurement2physicalvariable.xsl .measurement.xml.isvalid
	@XALAN@ -o physicalvariable.xsd $(srcdir)/measurement.xml $(srcdir)/measurement2physicalvariable.xsl



CLEANFILES = .measurement.xml.isvalid physicalvariable.xsd



# utility library
lib_LTLIBRARIES = libmbxmlutils.la
libmbxmlutils_ladir = $(includedir)/mbxmlutils
libmbxmlutils_la_CPPFLAGS = -I$(top_srcdir) $(MKOCTFILE_CFLAGS) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
libmbxmlutils_la_LIBADD = ../mbxmlutilshelper/libmbxmlutilshelper.la $(CASADI_LIBS) $(MKOCTFILE_LIBS) $(XERCESC_LIBS) $(FMATVEC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@
libmbxmlutils_la_SOURCES = octeval.cc preprocess.cc
libmbxmlutils_la_HEADERS = octeval.h preprocess.h

# program to convert a complex xml file to a plain xml file
bin_PROGRAMS = mbxmlutilspp
mbxmlutilspp_SOURCES = preprocessor.cc
mbxmlutilspp_CPPFLAGS = -I$(top_srcdir) $(MKOCTFILE_CFLAGS) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
mbxmlutilspp_LDADD = libmbxmlutils.la ../mbxmlutilshelper/libmbxmlutilshelper.la $(CASADI_LIBS) $(MKOCTFILE_LIBS) $(XERCESC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@



# generate deplib files for octave oct-files and casaid_interface.oct
if MBXMLUTILS_MINGW
  runlibdir = $(bindir)
else
  runlibdir = $(libdir)
endif
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
	if test $(MBXMLUTILS_CASADI_BIN)/casadi_interface.oct -nt $(runlibdir)/casadi_interface.oct.deplibs; then \
	  echo "Create dependency file for $(MBXMLUTILS_CASADI_BIN)/casadi_interface.oct"; \
	  cp -u $(MBXMLUTILS_CASADI_BIN)/casadi_interface.oct $(runlibdir)/casadi_interface.oct 2> /dev/null || \
	    echo "Copying skipped, maybe same file"; \
	  python $(prefix)/share/mbxmlutils/python/deplibs.py $(runlibdir)/casadi_interface.oct > \
	    $(runlibdir)/casadi_interface.oct.deplibs.tmp; \
	  mv -f $(runlibdir)/casadi_interface.oct.deplibs.tmp $(runlibdir)/casadi_interface.oct.deplibs; \
	fi
