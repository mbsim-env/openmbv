# We link libmbxmlutilstinyxml statically to avoid symbol conflicts with other projects
# also linking with TinyXML. This is required since we use a hacked version fo TinyXML.
# For Windows this does not work currently. Hence we build dynamicall in this case
# (But it should work on Windows)
# On Linux we must use the visibility hidden flag to enforce all symbols of TinyXML to
# be local to avoid symbol conflict.
# NOT WORKING CURRENTLY
#if MBXMLUTILS_MINGW
#else
#  STATIC = -static
#  VISIBILITYHIDDEN = -fvisibility=hidden
#endif
lib_LTLIBRARIES = libmbxmlutilstinyxml.la
libmbxmlutilstinyxml_la_LDFLAGS = $(STATIC)
libmbxmlutilstinyxml_la_CPPFLAGS = $(VISIBILITYHIDDEN)

INCLUDES = -DTIXML_USE_STL
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
