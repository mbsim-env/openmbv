# We link libmbxmlutilstinyxml statically to avoid symbol conflicts with other projects
# also linking with TinyXML. This is required since we use a hacked version fo TinyXML.
# For Windows this does not work currently. Hence we build dynamicall in this case
# (But it should work on Windows)
if MBXMLUTILS_MINGW
else
  STATIC = -static
endif
lib_LTLIBRARIES = libmbxmlutilstinyxml.la
libmbxmlutilstinyxml_la_LDFLAGS = $(STATIC)

INCLUDES = -DTIXML_USE_STL
libmbxmlutilstinyxml_la_SOURCES = tinynamespace.cpp \
  tinystr.cpp \
  tinyxml.cpp \
  tinyxmlerror.cpp \
  tinyxmlparser.cpp \
  getinstallpath.cc \
  last_write_time.cc

libmbxmlutilstinyxml_la_HEADERS = tinynamespace.h \
  tinystr.h \
  tinyxml.h \
  getinstallpath.h \
  last_write_time.h \
  casadiXML.h

libmbxmlutilstinyxml_ladir = $(includedir)/mbxmlutilstinyxml

EXTRA_DIST = readme.txt changes.txt
