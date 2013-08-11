SUBDIRS = . check

lib_LTLIBRARIES = libopenmbvcppinterface.la

libopenmbvcppinterface_ladir = $(includedir)/openmbvcppinterface

libopenmbvcppinterface_la_CPPFLAGS = -I$(top_srcdir) -DTIXML_USE_STL $(HDF5SERIE_CFLAGS)

libopenmbvcppinterface_la_LDFLAGS = $(HDF5SERIE_LIBS) $(MBXMLUTILSTINYXML_LIBS)

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
  arrow.cc \
  path.cc \
  coilspring.cc \
  cube.cc \
  ivbody.cc \
  cuboid.cc \
  spineextrusion.cc \
  nurbsdisk.cc

libopenmbvcppinterface_la_HEADERS = object.h \
  objectfactory.h\
  body.h \
  simpleparameter.h \
  dynamiccoloredbody.h \
  group.h \
  rigidbody.h \
  rigidbodycos.png \
  rigidbodycos.eps \
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
  nurbsdisk.h
