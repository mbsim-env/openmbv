SUBDIRS = . check

lib_LTLIBRARIES = libopenmbvcppinterface.la

libopenmbvcppinterface_ladir = $(includedir)/openmbvcppinterface

libopenmbvcppinterface_la_CPPFLAGS = -I$(top_srcdir) $(HDF5SERIE_CFLAGS) $(MBXMLUTILSHELPER_CFLAGS) $(FMATVEC_CFLAGS)

libopenmbvcppinterface_la_LDFLAGS = $(HDF5SERIE_LIBS) $(MBXMLUTILSHELPER_LIBS) $(FMATVEC_LIBS) -l@BOOST_SYSTEM_LIB@

libopenmbvcppinterface_la_SOURCES = object.cc \
  objectfactory.cc\
  body.cc \
  dynamiccoloredbody.cc \
  group.cc \
  rigidbody.cc \
  compoundrigidbody.cc \
  frame.cc \
  grid.cc \
  frustum.cc \
  sphere.cc \
  extrusion.cc \
  rotation.cc \
  invisiblebody.cc \
  polygonpoint.cc \
  arrow.cc \
  path.cc \
  coilspring.cc \
  cube.cc \
  ivbody.cc \
  cuboid.cc \
  spineextrusion.cc \
  nurbsdisk.cc \
  nurbscurve.cc \
  nurbssurface.cc \
  dynamicnurbssurface.cc \
  indexedfaceset.cc \
  dynamicindexedfaceset.cc

libopenmbvcppinterface_la_HEADERS = object.h \
  objectfactory.h\
  body.h \
  dynamiccoloredbody.h \
  group.h \
  rigidbody.h \
  rigidbodycos.svg \
  compoundrigidbody.h \
  frame.h \
  grid.h \
  frustum.h \
  sphere.h \
  extrusion.h \
  rotation.h \
  invisiblebody.h \
  polygonpoint.h \
  arrow.h \
  path.h \
  coilspring.h \
  cube.h \
  ivbody.h \
  cuboid.h \
  spineextrusion.h \
  nurbsdisk.h \
  nurbscurve.h \
  nurbssurface.h \
  dynamicnurbssurface.h \
  indexedfaceset.h \
  dynamicindexedfaceset.h
