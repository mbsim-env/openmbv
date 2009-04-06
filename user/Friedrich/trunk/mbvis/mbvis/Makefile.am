include $(top_srcdir)/autotroll.mk

bin_PROGRAMS = mbvis
#mbvis_CXXFLAGS = $(QT_CXXFLAGS) $(AM_CXXFLAGS)
#mbvis_CPPFLAGS = $(QT_CPPFLAGS) $(AM_CPPFLAGS)
mbvis_CPPFLAGS = $(QT_CPPFLAGS) -I../tinyxml $(COIN_CFLAGS) $(SOQT_CFLAGS) -DTIXML_USE_STL
mbvis_LDFLAGS  = -L../tinyxml $(COIN_LDFLAGS) $(QT_LDFLAGS) $(SOQT_LDFALGS)
mbvis_LDADD    = -ltinyxml $(COIN_LIBS) $(QT_LIBS) $(SOQT_LIBS)
mbvis_SOURCES = body.cc\
  cuboid.cc\
  frame.cc\
  group.cc\
  main.cc\
  mainwindow.cc\
  object.cc\
  objectfactory.cc\
  rigidbody.cc\
  mainwindow.moc.cc\
  object.moc.cc\
  rigidbody.moc.cc\
  cuboid.moc.cc\
  frame.moc.cc\
  group.moc.cc\
  body.moc.cc

BUILT_SOURCES = mainwindow.moc.cc\
  object.moc.cc\
  rigidbody.moc.cc\
  cuboid.moc.cc\
  frame.moc.cc\
  group.moc.cc\
  body.moc.cc
