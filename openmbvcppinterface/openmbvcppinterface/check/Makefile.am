check_PROGRAMS = testprog

TESTS = testprog.sh

testprog_SOURCES = testprog.cc

AM_CPPFLAGS = $(HDF5SERIE_CFLAGS) $(MBXMLUTILSHELPER_CFLAGS)

testprog_CXXFLAGS = -I$(top_srcdir)
testprog_LDADD = ../libopenmbvcppinterface.la $(HDF5SERIE_LIBS) $(MBXMLUTILSHELPER_LIBS)
