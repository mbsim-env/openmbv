include $(top_srcdir)/autotroll.mk

bin_PROGRAMS = mbvis
mbvis_CPPFLAGS = $(QT_CPPFLAGS) -I../tinyxml $(COIN_CFLAGS) $(SOQT_CFLAGS) -DTIXML_USE_STL $(HDF5SERIE_CFLAGS)
mbvis_LDFLAGS  = -L../tinyxml $(COIN_LDFLAGS) $(QT_LDFLAGS) $(SOQT_LDFALGS) $(HDF5SERIE_LDFLAGS)
mbvis_LDADD    = -ltinyxml $(COIN_LIBS) $(QT_LIBS) $(SOQT_LIBS) $(HDF5SERIE_LIBS)
mbvis_SOURCES = body.cc\
  cuboid.cc\
  frame.cc\
  group.cc\
  main.cc\
  mainwindow.cc\
  object.cc\
  objectfactory.cc\
  rigidbody.cc

nodist_mbvis_SOURCES=$(BUILT_SOURCES)

# C++ source file to build automatically (e.g. abc.moc.cc by abc.h)
BUILT_SOURCES = mbvis.qrc.cc\
  mainwindow.moc.cc\
  object.moc.cc\
  rigidbody.moc.cc\
  cuboid.moc.cc\
  frame.moc.cc\
  group.moc.cc\
  body.moc.cc\
  SoQtMyViewer.cc

noinst_HEADERS = mbvis.qrc\
  mainwindow.h\
  object.h\
  rigidbody.h\
  cuboid.h\
  frame.h\
  group.h\
  body.h\
  SoQtMyViewer.h
