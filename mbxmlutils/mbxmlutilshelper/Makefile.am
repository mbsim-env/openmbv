lib_LTLIBRARIES = libmbxmlutilshelper.la

# XML helper functions
libmbxmlutilshelper_la_CPPFLAGS = $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
libmbxmlutilshelper_la_LDFLAGS = $(UNWIND_LIB) $(FMATVEC_CFLAGS)
libmbxmlutilshelper_la_LIBADD = $(XERCESC_LIBS) $(FMATVEC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@ $(LIBDL)
libmbxmlutilshelper_la_SOURCES = dom.cc getinstallpath.cc utils.cc
libmbxmlutilshelper_la_HEADERS = dom.h getinstallpath.h utils.h casadiXML.h shared_library.h last_write_time.h toString.h getsharedlibpath.h getsharedlibpath_impl.h
libmbxmlutilshelper_ladir = $(includedir)/mbxmlutilshelper

w3cschemadir = @SCHEMADIR@/http___www_w3_org
dist_w3cschema_DATA = XInclude.xsd


# small program to validate xml files
bin_PROGRAMS = mbxmlutilsvalidate
mbxmlutilsvalidate_SOURCES = validate.cc
mbxmlutilsvalidate_CPPFLAGS = $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
mbxmlutilsvalidate_LDADD = libmbxmlutilshelper.la $(XERCESC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@


pythondir = @datadir@/@PACKAGE@/python
dist_python_DATA = deplibs.py deplibs.mk
