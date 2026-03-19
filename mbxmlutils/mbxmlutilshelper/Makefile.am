lib_LTLIBRARIES = libmbxmlutilshelper.la

if COND_WIN32
  MAYBE_WIN32_mbxmlutilsvalidate_OBJ = mbxmlutilsvalidate.o
  MAYBE_WIN32_mbxmlutilsprettyprint = mbxmlutilsprettyprint.o
endif

# windows rc rules
mbxmlutilsvalidate.o: $(srcdir)/mbxmlutilsvalidate.rc $(srcdir)/mbxmlutilsvalidate.manifest
	$(WINDRES) $< -o $@
mbxmlutilsprettyprint.o: $(srcdir)/mbxmlutilsprettyprint.rc $(srcdir)/mbxmlutilsprettyprint.manifest
	$(WINDRES) $< -o $@

# XML helper functions
libmbxmlutilshelper_la_CPPFLAGS = $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
libmbxmlutilshelper_la_LDFLAGS = $(UNWIND_LIB) $(FMATVEC_CFLAGS)
libmbxmlutilshelper_la_LIBADD = $(LDFLAGS_NOUNDEF) $(XERCESC_LIBS) $(FMATVEC_LIBS) -l@BOOST_REGEX_LIB@ -l@BOOST_FILESYSTEM_LIB@ $(LIBDL)
libmbxmlutilshelper_la_SOURCES = dom.cc simdutf.cpp utils.cc windows_signal_conversion.cc
libmbxmlutilshelper_la_HEADERS = dom.h simdutf.h utils.h windows_signal_conversion.h shared_library.h last_write_time.h thislinelocation.h
libmbxmlutilshelper_ladir = $(includedir)/mbxmlutilshelper

w3cschemadir = @datadir@/@PACKAGE@/schema/http___www_w3_org
dist_w3cschema_DATA = XInclude.xsd


bin_PROGRAMS = mbxmlutilsvalidate mbxmlutilsprettyprint

# small program to validate xml files
mbxmlutilsvalidate_SOURCES = validate.cc
mbxmlutilsvalidate_CPPFLAGS = $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
mbxmlutilsvalidate_LDADD = $(LDFLAGS_NOUNDEF) libmbxmlutilshelper.la $(XERCESC_LIBS) -l@BOOST_FILESYSTEM_LIB@ $(FMATVEC_LIBS) $(MAYBE_WIN32_mbxmlutilsvalidate_OBJ)

# small program to pretty-print xml files
mbxmlutilsprettyprint_SOURCES = prettyprint.cc
mbxmlutilsprettyprint_CPPFLAGS = $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
mbxmlutilsprettyprint_LDADD = $(LDFLAGS_NOUNDEF) libmbxmlutilshelper.la $(XERCESC_LIBS) -l@BOOST_FILESYSTEM_LIB@ $(FMATVEC_LIBS) $(MAYBE_WIN32_mbxmlutilsprettyprint_OBJ)


pythondir = @datadir@/@PACKAGE@/python
dist_python_DATA = deplibs.py deplibs.mk deplibs.cmake
