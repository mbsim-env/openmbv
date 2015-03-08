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
libmbxmlutils_la_LDFLAGS = -Wl,--disable-new-dtags,-rpath,\$$ORIGIN/../lib
libmbxmlutils_la_LIBADD = ../mbxmlutilshelper/libmbxmlutilshelper.la $(CASADI_LIBS) $(MKOCTFILE_LIBS) $(XERCESC_LIBS) $(FMATVEC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@
libmbxmlutils_la_SOURCES = octeval.cc preprocess.cc
libmbxmlutils_la_HEADERS = octeval.h preprocess.h

# program to convert a complex xml file to a plain xml file
bin_PROGRAMS = mbxmlutilspp
mbxmlutilspp_SOURCES = preprocessor.cc
mbxmlutilspp_CPPFLAGS = -I$(top_srcdir) $(MKOCTFILE_CFLAGS) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
mbxmlutilspp_LDFLAGS = -Wl,--disable-new-dtags,-rpath,\$$ORIGIN/../lib
mbxmlutilspp_LDADD = libmbxmlutils.la ../mbxmlutilshelper/libmbxmlutilshelper.la $(MKOCTFILE_LIBS) $(XERCESC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@
