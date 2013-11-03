lib_LTLIBRARIES = libmbxmlutilstinyxml.la
libmbxmlutilstinyxml_la_CPPFLAGS = -I$(top_srcdir) -DTIXML_USE_STL 
libmbxmlutilstinyxml_la_LDFLAGS = $(UNWIND_LIB)
libmbxmlutilstinyxml_la_LIBADD = -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@

libmbxmlutilstinyxml_la_SOURCES = tinynamespace.cpp \
  tinystr.cpp \
  tinyxml.cpp \
  tinyxmlerror.cpp \
  tinyxmlparser.cpp \
  getinstallpath.cc \
  utils.cc \
  last_write_time.cc

libmbxmlutilstinyxml_la_HEADERS = tinynamespace.h \
  tinystr.h \
  tinyxml.h \
  getinstallpath.h \
  last_write_time.h \
  utils.h \
  casadiXML.h

libmbxmlutilstinyxml_ladir = $(includedir)/mbxmlutilstinyxml

EXTRA_DIST = readme.txt changes.txt
