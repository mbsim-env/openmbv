include $(top_srcdir)/qt.mk

if COND_WIN32
  MAYBE_WIN32_SRC = openmbv.rc
  LIBGLU=-lglu32 -lopengl32
else
  LIBGLU=-lGLU -lGL
  MAYBE_SIGWATCH_SRC = qt-unix-signals/sigwatch.cpp
  MAYBE_SIGWATCH_MOC = qt-unix-signals/sigwatch.moc.cc
endif

licdir = @datadir@/openmbv/qt-unix-signals
lic_DATA = qt-unix-signals/LICENCE

lib_LTLIBRARIES = libopenmbv.la
libopenmbv_la_CPPFLAGS = $(QT_CFLAGS) $(COIN_CFLAGS) $(OPENMBVCPPINTERFACE_CFLAGS) $(SOQT_CFLAGS) $(HDF5SERIE_CFLAGS) $(QWT_CFLAGS)
libopenmbv_la_LIBADD  = $(LIBGLU) $(COIN_LIBS) $(OPENMBVCPPINTERFACE_LIBS) $(QT_LIBS) $(SOQT_LIBS) $(HDF5SERIE_LIBS) $(QWT_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@
libopenmbv_la_SOURCES = $(MAYBE_SIGWATCH_SRC) \
  body.cc\
  dynamiccoloredbody.cc\
  cuboid.cc\
  cube.cc\
  extrusion.cc\
  rotation.cc\
  sphere.cc\
  invisiblebody.cc\
  frustum.cc\
  ivbody.cc\
  frame.cc\
  grid.cc\
  group.cc\
  mainwindow.cc\
  exportdialog.cc\
  object.cc\
  utils.cc\
  edgecalculation.cc\
  path.cc\
  arrow.cc\
  objectfactory.cc\
  rigidbody.cc\
  compoundrigidbody.cc\
  SoQtMyViewer.cc \
  SoTransposeEngine.cc \
  spineextrusion.cc \
  coilspring.cc \
  nurbsdisk.cc \
  nurbscurve.cc \
  nurbssurface.cc \
  dynamicnurbscurve.cc \
  dynamicnurbssurface.cc \
  pointset.cc \
  indexedlineset.cc \
  indexedfaceset.cc \
  flexiblebody.cc \
  dynamicpointset.cc \
  dynamicindexedlineset.cc \
  dynamicindexedfaceset.cc \
  cylindricalgear.cc \
  cylinder.cc \
  rack.cc \
  bevelgear.cc \
  planargear.cc \
  editors.cc \
  IndexedTesselationFace.cc \
  abstractviewfilter.cc \
  QTripleSlider.cc

nodist_libopenmbv_la_SOURCES=$(QT_BUILT_SOURCES)

# C++ source file to build automatically (e.g. abc.moc.cc by abc.h)
QT_BUILT_SOURCES = \
  mainwindow.moc.cc\
  exportdialog.moc.cc\
  object.moc.cc\
  edgecalculation.moc.cc\
  path.moc.cc\
  coilspring.moc.cc\
  spineextrusion.moc.cc \
  nurbsdisk.moc.cc \
  nurbscurve.moc.cc \
  nurbssurface.moc.cc \
  dynamicnurbscurve.moc.cc \
  dynamicnurbssurface.moc.cc \
  pointset.moc.cc \
  indexedlineset.moc.cc \
  indexedfaceset.moc.cc \
  flexiblebody.moc.cc \
  dynamicpointset.moc.cc \
  dynamicindexedlineset.moc.cc \
  dynamicindexedfaceset.moc.cc \
  cylindricalgear.moc.cc \
  cylinder.moc.cc \
  rack.moc.cc \
  bevelgear.moc.cc \
  planargear.moc.cc \
  editors.moc.cc \
  arrow.moc.cc\
  rigidbody.moc.cc\
  compoundrigidbody.moc.cc\
  cuboid.moc.cc\
  cube.moc.cc\
  extrusion.moc.cc\
  rotation.moc.cc\
  sphere.moc.cc\
  invisiblebody.moc.cc\
  frustum.moc.cc\
  ivbody.moc.cc\
  frame.moc.cc\
  grid.moc.cc\
  group.moc.cc\
  body.moc.cc\
  dynamiccoloredbody.moc.cc \
  QTripleSlider.moc.cc
BUILT_SOURCES = $(QT_BUILT_SOURCES) $(MAYBE_SIGWATCH_MOC)



bin_PROGRAMS = openmbv
openmbv_SOURCES = $(MAYBE_WIN32_SRC)\
  main.cc
openmbv_CPPFLAGS = $(QT_CFLAGS) $(COIN_CFLAGS) $(SOQT_CFLAGS) $(OPENMBVCPPINTERFACE_CFLAGS) $(QWT_CFLAGS)
openmbv_LDADD    = libopenmbv.la $(OPENMBVCPPINTERFACE_LIBS) $(QT_LIBS) $(SOQT_LIBS) $(QWT_LIBS) -l@BOOST_SYSTEM_LIB@



libopenmbv_ladir = $(includedir)/openmbv
libopenmbv_la_HEADERS = mainwindow.h\
  SoTransposeEngine.h \
  SoSpecial.h\
  object.h\
  utils.h\
  edgecalculation.h\
  path.h\
  arrow.h\
  rigidbody.h\
  compoundrigidbody.h\
  exportdialog.h\
  cuboid.h\
  cube.h\
  extrusion.h\
  rotation.h\
  sphere.h\
  invisiblebody.h\
  frustum.h\
  ivbody.h\
  frame.h\
  grid.h\
  group.h\
  body.h\
  dynamiccoloredbody.h\
  SoQtMyViewer.h \
  spineextrusion.h \
  coilspring.h \
  nurbsdisk.h \
  nurbscurve.h \
  nurbssurface.h \
  dynamicnurbscurve.h \
  dynamicnurbssurface.h \
  pointset.h \
  indexedlineset.h \
  indexedfaceset.h \
  flexiblebody.h \
  dynamicpointset.h \
  dynamicindexedlineset.h \
  dynamicindexedfaceset.h \
  cylindricalgear.h \
  cylinder.h \
  rack.h \
  bevelgear.h \
  planargear.h \
  editors.h \
  IndexedTesselationFace.h \
  abstractviewfilter.h \
  QTripleSlider.h

icondir = @datadir@/openmbv/icons
dist_icon_DATA = openmbv.svg \
  cuboid.svg \
  cube.svg \
  coilspring.svg \
  extrusion.svg \
  rotation.svg \
  frame.svg \
  group.svg \
  h5file.svg \
  unloadfile.svg \
  addfile.svg \
  savefile.svg \
  newfile.svg \
  quit.svg \
  viewall.svg \
  camera.svg \
  camerabody.svg \
  axialview.svg \
  topview.svg \
  bottomview.svg \
  frontview.svg \
  backview.svg \
  leftview.svg \
  rightview.svg \
  spaceview.svg \
  isometricview.svg \
  dimetricview.svg \
  rotateview.svg \
  drawobject.svg \
  bbox.svg \
  outline.svg \
  shilouetteedge.svg \
  filled.svg \
  lines.svg \
  points.svg \
  localframe.svg \
  referenceframe.svg \
  sphere.svg \
  grid.svg \
  invisiblebody.svg \
  frustum.svg \
  ivbody.svg \
  play.svg \
  lastframe.svg \
  stop.svg \
  exportimgsequence.svg \
  exportimg.svg \
  exportiv.svg \
  path.svg \
  arrow.svg \
  centerballdragger.svg \
  bgcolor.svg \
  loadwst.svg \
  savewst.svg \
  loadcamera.svg \
  savecamera.svg \
  engdrawing.svg \
  help.svg \
  compoundrigidbody.svg \
  olselinewidth.svg \
  olsecolor.svg \
  guihelp.html \
  complexitytype.svg \
  complexityvalue.svg \
  newobject.svg \
  deleteobject.svg \
  cover.png \
  knurl.png

# guihelp.html
docdir = @datadir@/openmbv/doc
dist_doc_DATA = guihelp.html

# plugins
pluginsiconenginesdir = @bindir@/iconengines
pluginsplatformsdir = @bindir@/platforms
pluginsstylesdir = @bindir@/styles
pluginsplatformthemesdir = @bindir@/platformthemes
pluginsiconengines_DATA = @QT_PLUGINS@/iconengines/*qsvgicon*
if COND_WIN32
pluginsplatforms_DATA = @QT_PLUGINS@/platforms/qwindows.dll
pluginsstyles_DATA = @QT_PLUGINS@/styles/qwindowsvistastyle.dll
else
pluginsplatforms_DATA = @QT_PLUGINS@/platforms/libqxcb.so
pluginsstyles_DATA = @QT_PLUGINSSTYLES_DATA@
pluginsplatformthemes_DATA = @QT_PLUGINSPLATFORMTHEMES_DATA@
pluginsxcbglintegrationsdir = @bindir@/xcbglintegrations
pluginsxcbglintegrations_DATA = @QT_PLUGINS@/xcbglintegrations/*qxcb*
endif

# freedesktop.org
freedesktoporgdir = @datadir@/mbsim-env/freedesktop.org
windowsdir = @datadir@/mbsim-env/windows
freedesktoporg_DATA = mbsim-env.de.openmbv.xml mbsim-env.de.openmbv.desktop mimeapps-openmbv.list openmbv.svg
windows_DATA = createLnk.vbs mbsim-env.de.openmbv.source.bat
bin_SCRIPTS = desktopIntegration.sh desktopIntegration.bat
