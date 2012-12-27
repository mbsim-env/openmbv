check_PROGRAMS = testprog

TESTS = testprog

testprog_SOURCES = testprog.cc

INCLUDES = $(HDF5SERIE_CFLAGS)

testprog_CXXFLAGS = -DTIXML_USE_STL 
testprog_LDADD = -lopenmbvcppinterface $(HDF5SERIE_LIBS) $(MBXMLUTILSTINYXML_LIBS)
