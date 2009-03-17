check_PROGRAMS = testprog

TESTS = testprog

testprog_SOURCES = testprog.cc

INCLUDES = $(HDF5SERIE_CFLAGS)

testprog_LDFLAGS = -L..
testprog_LDADD = -lamviscppinterface $(HDF5SERIE_LIBS)
