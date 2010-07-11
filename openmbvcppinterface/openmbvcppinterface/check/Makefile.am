check_PROGRAMS = testprog

TESTS = testprog

testprog_SOURCES = testprog.cc

INCLUDES = $(HDF5SERIE_CFLAGS)

testprog_CXXFLAGS = -DTIXML_USE_STL 
testprog_LDFLAGS = -L.. -L../../openmbvcppinterfacetinyxml
testprog_LDADD = -lopenmbvcppinterface $(HDF5SERIE_LIBS) -lopenmbvcppinterfacetinyxml
