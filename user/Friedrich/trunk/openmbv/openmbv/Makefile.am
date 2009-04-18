include $(top_srcdir)/autotroll.mk

if COND_QWT
  MAYBE_QWT_CFLAGS = $(QWT_CFLAGS)
  MAYBE_QWT_LDFLAGS_ = $(QWT_LDFLAGS_)
  MAYBE_QWT_LIBS = $(QWT_LIBS)
endif

bin_PROGRAMS = openmbv
openmbv_CPPFLAGS = $(QT_CPPFLAGS) -I../tinyxml $(COIN_CFLAGS) $(SOQT_CFLAGS) -DTIXML_USE_STL $(HDF5SERIE_CFLAGS) $(MAYBE_QWT_CFLAGS)
openmbv_LDFLAGS  = -L../tinyxml $(COIN_LDFLAGS) $(QT_LDFLAGS) $(SOQT_LDFALGS) $(HDF5SERIE_LDFLAGS) $(MAYBE_QWT_LDFLAGS_)
openmbv_LDADD    = -ltinyxml $(COIN_LIBS) $(QT_LIBS) $(SOQT_LIBS) $(HDF5SERIE_LIBS) $(MAYBE_QWT_LIBS)
openmbv_SOURCES = body.cc\
  cuboid.cc\
  cube.cc\
  sphere.cc\
  invisiblebody.cc\
  frustum.cc\
  ivbody.cc\
  frame.cc\
  group.cc\
  main.cc\
  mainwindow.cc\
  exportdialog.cc\
  object.cc\
  objectfactory.cc\
  rigidbody.cc

nodist_openmbv_SOURCES=$(BUILT_SOURCES)

# C++ source file to build automatically (e.g. abc.moc.cc by abc.h)
BUILT_SOURCES = openmbv.qrc.cc\
  mainwindow.moc.cc\
  exportdialog.moc.cc\
  object.moc.cc\
  rigidbody.moc.cc\
  cuboid.moc.cc\
  cube.moc.cc\
  sphere.moc.cc\
  invisiblebody.moc.cc\
  frustum.moc.cc\
  ivbody.moc.cc\
  frame.moc.cc\
  group.moc.cc\
  body.moc.cc\
  SoQtMyViewer.cc

noinst_HEADERS = openmbv.qrc\
  mainwindow.h\
  object.h\
  rigidbody.h\
  exportdialog.h\
  cuboid.h\
  cube.h\
  sphere.h\
  invisiblebody.h\
  frustum.h\
  ivbody.h\
  frame.h\
  group.h\
  body.h\
  SoQtMyViewer.h
