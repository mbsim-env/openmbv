SUBDIRS = . check

lib_LTLIBRARIES = libamviscppinterface.la

libamviscppinterface_ladir = $(includedir)/amviscppinterface

INCLUDES = $(HDF5SERIE_CFLAGS)

libamviscppinterface_la_SOURCES = object.cc \
  body.cc \
  group.cc \
  rigidbody.cc \
  cuboid.cc

libamviscppinterface_la_HEADERS = object.h \
  body.h \
  group.h \
  rigidbody.h \
  cuboid.h
