check_PROGRAMS = testprog

TESTS = testprog

testprog_SOURCES = testprog.cc

INCLUDES = $(HDF5SERIE_CFLAGS)

testprog_CXXFLAGS = -I$(top_srcdir)
testprog_LDADD = ../libopenmbvcppinterface.la $(HDF5SERIE_LIBS)
