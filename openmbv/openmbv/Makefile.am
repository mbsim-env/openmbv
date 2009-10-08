include $(top_srcdir)/qt.mk

if COND_QWT
  MAYBE_QWT_CFLAGS = $(QWT_CFLAGS)
  MAYBE_QWT_LIBS = $(QWT_LIBS)
endif

if COND_WIN32
  MAYBE_WIN32_SRC = openmbv.rc
  LIBGLU=-lglu32
else
  LIBGLU=-lGLU
endif

bin_PROGRAMS = openmbv
openmbv_CPPFLAGS = $(QT_CFLAGS) -I../openmbvtinyxml/tinyxml-src $(COIN_CFLAGS) $(SOQT_CFLAGS) -DTIXML_USE_STL $(HDF5SERIE_CFLAGS) $(MAYBE_QWT_CFLAGS)
openmbv_LDFLAGS  = -L../openmbvtinyxml 
openmbv_LDADD    = $(LIBGLU) -lopenmbvtinyxml $(COIN_LIBS) $(QT_LIBS) $(SOQT_LIBS) $(HDF5SERIE_LIBS) $(MAYBE_QWT_LIBS)
openmbv_SOURCES = $(MAYBE_WIN32_SRC)\
  body.cc\
  dynamiccoloredbody.cc\
  cuboid.cc\
  cube.cc\
  extrusion.cc\
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
  path.cc\
  arrow.cc\
  objbody.cc\
  objectfactory.cc\
  rigidbody.cc\
  compoundrigidbody.cc\
  SoQtMyViewer.cc \
  SoTransposeEngine.cc \
  spineextrusion.cc \
  coilspring.cc \
  nurbsdisk.cc

nodist_openmbv_SOURCES=$(QT_BUILT_SOURCES)

# C++ source file to build automatically (e.g. abc.moc.cc by abc.h)
QT_BUILT_SOURCES = openmbv.qrc.cc\
  mainwindow.moc.cc\
  exportdialog.moc.cc\
  object.moc.cc\
  path.moc.cc\
  coilspring.moc.cc\
  spineextrusion.moc.cc \
  nurbsdisk.moc.cc \
  arrow.moc.cc\
  objbody.moc.cc\
  rigidbody.moc.cc\
  compoundrigidbody.moc.cc\
  cuboid.moc.cc\
  cube.moc.cc\
  extrusion.moc.cc\
  sphere.moc.cc\
  invisiblebody.moc.cc\
  frustum.moc.cc\
  ivbody.moc.cc\
  frame.moc.cc\
  group.moc.cc\
  body.moc.cc\
  dynamiccoloredbody.moc.cc

noinst_HEADERS = openmbv.qrc\
  mainwindow.h\
  object.h\
  path.h\
  arrow.h\
  objbody.h\
  rigidbody.h\
  compoundrigidbody.h\
  exportdialog.h\
  cuboid.h\
  cube.h\
  extrusion.h\
  sphere.h\
  invisiblebody.h\
  frustum.h\
  ivbody.h\
  frame.h\
  group.h\
  body.h\
  dynamiccoloredbody.h\
  SoSpecial.h\
  SoQtMyViewer.h \
  SoTransposeEngine.h \
  spineextrusion.h \
  coilspring.h \
  nurbsdisk.h

# BEGIN generate openmbv.qrc from openbv.qrc.myin
# Subst PNGFILESTOINCLUDE with a <file> element for each png file in ../doc/*.png
openmbv.qrc: openmbv.qrc.in_sed ../doc/http___openmbv_berlios_de_OpenMBV/
	FILES=""; \
	for F in $$(ls -1 ../doc/http___openmbv_berlios_de_OpenMBV/*.png); do \
	  FILES="$$FILES<file alias=\"http___openmbv_berlios_de_OpenMBV/$$(basename $$F)\">$$F</file>"; \
	done; \
	for F in $$(ls -1 $$(pkg-config mbxmlutils --variable DOCDIR)/http___openmbv_berlios_de_MBXMLUtils_physicalvariable/*.png $$(pkg-config mbxmlutils --variable DOCDIR)/http___openmbv_berlios_de_MBXMLUtils_physicalvariable/index.xhtml); do \
	  FILES="$$FILES<file alias=\"http___openmbv_berlios_de_MBXMLUtils_physicalvariable/$$(basename $$F)\">$$F</file>"; \
	done; \
	sed -re "s|PNGFILESTOINCLUDE|$$FILES|" openmbv.qrc.in_sed > openmbv.qrc
# END generate openmbv.qrc from openbv.qrc.myin

# some dependencies for openmbv.qrc.cc
openmbv.qrc.cc: openmbv.qrc ../doc/http___openmbv_berlios_de_OpenMBV/index.xhtml

# generate dummy documentation if not exist
../doc/http___openmbv_berlios_de_OpenMBV/index.xhtml:
	echo "WARNING! XML docu do not exist. Creating empty dummy docu!"
	mkdir -p ../doc/http___openmbv_berlios_de_OpenMBV
	touch -r ../schema/schema/openmbv.xsd.in ../doc/http___openmbv_berlios_de_OpenMBV/index.xhtml
