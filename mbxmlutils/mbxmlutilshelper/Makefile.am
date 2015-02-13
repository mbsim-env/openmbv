if MBXMLUTILS_MINGW
  DL_LIB = 
else
  DL_LIB = -ldl
endif
lib_LTLIBRARIES = libmbxmlutilshelper.la
libmbxmlutilshelper_la_CPPFLAGS = $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
libmbxmlutilshelper_la_LDFLAGS = $(UNWIND_LIB) $(FMATVEC_CFLAGS) -Wl,-rpath,\$$ORIGIN/../lib
libmbxmlutilshelper_la_LIBADD = $(XERCESC_LIBS) $(FMATVEC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@ $(DL_LIB)

libmbxmlutilshelper_la_SOURCES = dom.cc getinstallpath.cc last_write_time.cc utils.cc shared_library.cc

libmbxmlutilshelper_la_HEADERS = dom.h getinstallpath.h last_write_time.h utils.h casadiXML.h shared_library.h

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
