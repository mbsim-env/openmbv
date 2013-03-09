SUBDIRS = octave check

# distribute and install
schemadir = @datadir@/@PACKAGE@/schema/http___openmbv_berlios_de_MBXMLUtils
xmldir = @datadir@/@PACKAGE@/xml
dist_schema_DATA = physicalvariable.xsd parameter.xsd xml.xsd xhtml1-transitional.xsd
dist_xml_DATA = measurement.xml





# validate measurement.xml
.measurement.xml.isvalid: measurement.xml measurement.xsd
	rm -f .measurement.xml.isvalid
	@XMLLINT@ --noout --schema measurement.xsd measurement.xml && touch .measurement.xml.isvalid




# generate physicalvariable.xsd
physicalvariable.xsd: measurement.xml measurement2physicalvariable.xsl .measurement.xml.isvalid
	@XSLTPROC@ --output physicalvariable.xsd measurement2physicalvariable.xsl measurement.xml
# link some files required for the "check" target to the "check" directory
	test -d check/http___openmbv_berlios_de_MBXMLUtils || mkdir check/http___openmbv_berlios_de_MBXMLUtils
	(cd check/http___openmbv_berlios_de_MBXMLUtils; \
	ln -fs ../../physicalvariable.xsd .; \
	ln -fs ../../xml.xsd .; \
	ln -fs ../../parameter.xsd .)



CLEANFILES = .measurement.xml.isvalid physicalvariable.xsd



# utility library
lib_LTLIBRARIES = libmbxmlutils.la
libmbxmlutils_ladir = $(includedir)/mbxmlutils
libmbxmlutils_la_CPPFLAGS = -DTIXML_USE_STL $(MKOCTFILE_CFLAGS)
libmbxmlutils_la_LDFLAGS = -L../mbxmlutilstinyxml -Wl,-rpath -Wl,\$$ORIGIN/../lib
libmbxmlutils_la_LIBADD = ../mbxmlutilstinyxml/libmbxmlutilstinyxml.la $(MKOCTFILE_LIBS)
libmbxmlutils_la_SOURCES = utils.cc
libmbxmlutils_la_HEADERS = utils.h

# program to convert a complex xml file to a plain xml file
bin_PROGRAMS = mbxmlutilspp
mbxmlutilspp_SOURCES = preprocessor.cc
mbxmlutilspp_CPPFLAGS = -DTIXML_USE_STL $(LIBXML2_CFLAGS) $(MKOCTFILE_CFLAGS)
mbxmlutilspp_LDFLAGS = -Wl,-rpath -Wl,\$$ORIGIN/../lib
mbxmlutilspp_LDADD = $(LIBXML2_LIBS) $(MKOCTFILE_LIBS) libmbxmlutils.la ../mbxmlutilstinyxml/libmbxmlutilstinyxml.la -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@
