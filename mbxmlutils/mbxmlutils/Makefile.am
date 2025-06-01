if COND_WIN32
  MAYBE_WIN32_mbxmlutilspp_OBJ = mbxmlutilspp.o
  runlibdir = $(bindir)
  shext = .dll
else
  runlibdir = $(libdir)
  shext = .so
endif

# windows rc rules
mbxmlutilspp.o: $(srcdir)/mbxmlutilspp.rc $(srcdir)/mbxmlutilspp.manifest
	$(WINDRES) $< -o $@

SUBDIRS = octave python

# distribute and install
mbxmlutilsschemadir = @datadir@/@PACKAGE@/schema/http___www_mbsim-env_de_MBXMLUtils
xmldir = @datadir@/@PACKAGE@/xml
dist_mbxmlutilsschema_DATA = types.xsd embed.xsd parameter.xsd mbxmlutils.xsd mbxmlutilsCatalog.xml
dist_xml_DATA = measurement.xml



# validate measurement.xml
TESTS = measurement.sh



# generate types.xsd
types.xsd: $(srcdir)/measurement.xml $(srcdir)/measurement2physicalvariable.xsl
	@XALAN@ -o types.xsd $(srcdir)/measurement.xml $(srcdir)/measurement2physicalvariable.xsl



lib_LTLIBRARIES = libpycppwrapper.la libmbxmlutils.la libmbxmlutils-eval-octave.la libmbxmlutils-eval-xmlflat.la libmbxmlutils-eval-global-python.la
module_LTLIBRARIES = fmatvec_symbolic_swig_octave.la registerPath.la getOriginalFilename.la
moduledir = $(runlibdir)

# utility library
libmbxmlutils_ladir = $(includedir)/mbxmlutils
libmbxmlutils_la_CPPFLAGS = -I$(top_srcdir) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
libmbxmlutils_la_LIBADD = $(LDFLAGS_NOUNDEF) ../mbxmlutilshelper/libmbxmlutilshelper.la $(XERCESC_LIBS) $(FMATVEC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@ $(LIBDL)
libmbxmlutils_la_SOURCES = eval.cc preprocess.cc
libmbxmlutils_la_HEADERS = eval.h preprocess.h

# octave evaluator
libmbxmlutils_eval_octave_ladir = $(includedir)/mbxmlutils
libmbxmlutils_eval_octave_la_CPPFLAGS = -I$(abs_top_srcdir) $(MKOCTFILE_CFLAGS) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS) $(NOWARNING)
libmbxmlutils_eval_octave_la_LIBADD = $(LDFLAGS_NOUNDEF) $(MKOCTFILE_LIBS) libmbxmlutils.la ../mbxmlutilshelper/libmbxmlutilshelper.la $(XERCESC_LIBS) $(FMATVEC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@
libmbxmlutils_eval_octave_la_LDFLAGS = -avoid-version
if !COND_WIN32
libmbxmlutils_eval_octave_la_LDFLAGS += -Wl,-rpath,$(MKOCTFILE_LIBDIR)
endif
libmbxmlutils_eval_octave_la_SOURCES = octeval.cc
libmbxmlutils_eval_octave_la_HEADERS = octeval.h
# octave registerPath
registerPath_la_CPPFLAGS = -I$(abs_top_srcdir) $(MKOCTFILE_CFLAGS) $(NOWARNING)
registerPath_la_LIBADD = $(LDFLAGS_NOUNDEF) $(MKOCTFILE_LIBS) libmbxmlutils.la
registerPath_la_LDFLAGS = -module -shrext .oct -avoid-version
if !COND_WIN32
registerPath_la_LDFLAGS += -Wl,-rpath,$(MKOCTFILE_LIBDIR)
endif
registerPath_la_SOURCES = octave_registerPath.cc
# octave getOriginalFilename
getOriginalFilename_la_CPPFLAGS = -I$(abs_top_srcdir) $(MKOCTFILE_CFLAGS) $(NOWARNING)
getOriginalFilename_la_LIBADD = $(LDFLAGS_NOUNDEF) $(MKOCTFILE_LIBS) libmbxmlutils.la -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@
getOriginalFilename_la_LDFLAGS = -module -shrext .oct -avoid-version
if !COND_WIN32
getOriginalFilename_la_LDFLAGS += -Wl,-rpath,$(MKOCTFILE_LIBDIR)
endif
getOriginalFilename_la_SOURCES = octave_getOriginalFilename.cc

# xmlflat dummy evaluator
libmbxmlutils_eval_xmlflat_ladir = $(includedir)/mbxmlutils
libmbxmlutils_eval_xmlflat_la_CPPFLAGS = -I$(top_srcdir) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
libmbxmlutils_eval_xmlflat_la_LIBADD = $(LDFLAGS_NOUNDEF) libmbxmlutils.la ../mbxmlutilshelper/libmbxmlutilshelper.la $(FMATVEC_LIBS) $(XERCESC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@
libmbxmlutils_eval_xmlflat_la_LDFLAGS = -avoid-version
libmbxmlutils_eval_xmlflat_la_SOURCES = xmlflateval.cc
libmbxmlutils_eval_xmlflat_la_HEADERS = xmlflateval.h

# python c++ wrapper librarey
libpycppwrapper_ladir = $(includedir)/mbxmlutils
libpycppwrapper_la_CPPFLAGS = $(PYTHON_CFLAGS) -D_hypot=hypot # -D_hypot is a MSVS/python bug workaround
libpycppwrapper_la_LIBADD = $(PYTHON_LIBS_OPT) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@ # do not link with PYTHON_LIBS and do not add LDFLAGS_NOUNDEF
libpycppwrapper_la_SOURCES = pycppwrapper.cc
libpycppwrapper_la_HEADERS = pycppwrapper.h
# python part of the utility library
libmbxmlutils_eval_global_python_ladir = $(includedir)/mbxmlutils
libmbxmlutils_eval_global_python_la_CPPFLAGS = -I$(top_srcdir) $(PYTHON_CFLAGS) -I$(PYTHON_LIBDIR)/$(PYTHON_PACKAGES)/numpy/core/include $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS) -D_hypot=hypot # -D_hypot is a MSVS/python bug workaround
libmbxmlutils_eval_global_python_la_LIBADD = $(PYTHON_LIBS_OPT) libmbxmlutils.la ./libpycppwrapper.la ../mbxmlutilshelper/libmbxmlutilshelper.la $(FMATVEC_LIBS) $(XERCESC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@ # do not link with PYTHON_LIBS and do not add LDFLAGS_NOUNDEF
libmbxmlutils_eval_global_python_la_LDFLAGS = -avoid-version
libmbxmlutils_eval_global_python_la_SOURCES = pyeval.cc
libmbxmlutils_eval_global_python_la_HEADERS = pyeval.h

BUILT_SOURCES = fmatvec_symbolic_swig_octave.cc

# program to convert a complex xml file to a plain xml file
bin_PROGRAMS = mbxmlutilspp
mbxmlutilspp_SOURCES = preprocessor.cc
mbxmlutilspp_CPPFLAGS = -I$(top_srcdir) $(XERCESC_CFLAGS) $(FMATVEC_CFLAGS)
mbxmlutilspp_LDADD = $(LDFLAGS_NOUNDEF) libmbxmlutils.la ../mbxmlutilshelper/libmbxmlutilshelper.la $(XERCESC_LIBS) -l@BOOST_FILESYSTEM_LIB@ -l@BOOST_SYSTEM_LIB@ $(FMATVEC_LIBS) $(LIBDL) $(MAYBE_WIN32_mbxmlutilspp_OBJ)



.DELETE_ON_ERROR: # swig -Werror requires this
fmatvec_symbolic_swig_octave.cc: $(srcdir)/fmatvec_symbolic_swig_octave.i
	rm -f fmatvec_symbolic_swig_octave.cc
	$(swig) -octave -MP -MD -Werror -c++ -globals . -o fmatvec_symbolic_swig_octave.cc $(FMATVEC_CFLAGS) $(srcdir)/fmatvec_symbolic_swig_octave.i
	sed -i -re "s/\<is_object\>/isobject/g" fmatvec_symbolic_swig_octave.cc || exit 0 # WORKAROUND for SWIG bug
-include $(builddir)/fmatvec_symbolic_swig_octave.d

fmatvec_symbolic_swig_octave_la_SOURCES = fmatvec_symbolic_swig_octave.cc
fmatvec_symbolic_swig_octave_la_LDFLAGS = -module -shrext .oct -avoid-version
if !COND_WIN32
fmatvec_symbolic_swig_octave_la_LDFLAGS += -Wl,-rpath,$(MKOCTFILE_LIBDIR)
endif
fmatvec_symbolic_swig_octave_la_CPPFLAGS = $(MKOCTFILE_CFLAGS) $(FMATVEC_CFLAGS) -Wno-deprecated-declarations -Wno-error=unused-function $(NOWARNING) -DSWIG_OCTAVE_NO_SEGFAULT_HACK -D_USE_MATH_DEFINES
fmatvec_symbolic_swig_octave_la_LIBADD = $(LDFLAGS_NOUNDEF) $(MKOCTFILE_LIBS) $(FMATVEC_LIBS)



CLEANFILES = types.xsd fmatvec_symbolic_swig_octave.cc fmatvec_symbolic_swig_octave.d



# generate deplib files for octave oct-files, fmatvec_symbolic_swig_octave, mbxmlutils-eval-octave and registerPath/getOriginalFilename
install-exec-local: install-moduleLTLIBRARIES
	set -e; \
	ANYFAILED=0; \
	for F in $(runlibdir)/fmatvec_symbolic_swig_octave.oct $(runlibdir)/libmbxmlutils-eval-octave$(shext) $(runlibdir)/registerPath.oct $(runlibdir)/getOriginalFilename.oct $(runlibdir)/libmbxmlutils-eval-global-python$(shext); do \
	  if test $$F -nt $$F.deplibs -o ! -e $$F; then \
	    echo "Create dependency file for $$F"; \
	    python3 $(prefix)/share/mbxmlutils/python/deplibs.py $$F > $$F.deplibs || ANYFAILED=1; \
	  fi; \
	done; \
	if test $$ANYFAILED = 1; then \
	  echo "At least one deplibs command failed, see above."; \
	  exit 1; \
	fi;
	set -e; \
	mkdir -p $(prefix)/share/deplibs; \
	ANYFAILED=0; \
	for i in $(MKOCTFILE_LIBDIR)/oct/*/*.oct \
	         $(PYTHON_LIBDIR2)/*.so \
	         $(PYTHON_LIBDIR2)/*.dll \
	         $(PYTHON_LIBDIR2)/*.pyd \
	         $(PYTHON_LIBDIR)/$(PYTHON_PACKAGES)/numpy/*/*.so* \
	         $(PYTHON_LIBDIR)/$(PYTHON_PACKAGES)/numpy/*/*.pyd \
	         $(PYTHON_LIBDIR)/$(PYTHON_PACKAGES)/numpy/.libs/*.dll; do \
	  test $$(basename $$i) == __init_gnuplot__.oct && continue; \
	  if test $$i -nt $(prefix)/share/deplibs/$$(basename $$i).deplibs; then \
	    echo "Create dependency file and wrapper file for $$i"; \
	    python3 $(prefix)/share/mbxmlutils/python/deplibs.py $$i > $(prefix)/share/deplibs/$$(basename $$i).deplibs || ANYFAILED=1; \
	  fi; \
	done; \
	if test $$ANYFAILED = 1; then \
	  echo "At least one deplibs command failed, see above."; \
	  exit 1; \
	fi;
