lib_LTLIBRARIES = libmbxmlutilshelper.la
libmbxmlutilshelper_la_CPPFLAGS = $(XERCESC_CFLAGS)
libmbxmlutilshelper_la_LDFLAGS = $(UNWIND_LIB)
libmbxmlutilshelper_la_LIBADD = $(XERCESC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@ -l@BOOST_LOCALE_LIB@

libmbxmlutilshelper_la_SOURCES = dom.cc getinstallpath.cc last_write_time.cc utils.cc

libmbxmlutilshelper_la_HEADERS = dom.h getinstallpath.h last_write_time.h utils.h casadiXML.h

libmbxmlutilshelper_ladir = $(includedir)/mbxmlutilshelper

w3cschemadir = @SCHEMADIR@/http___www_w3_org
dist_w3cschema_DATA = XInclude.xsd xml.xsd xhtml1-transitional.xsd
