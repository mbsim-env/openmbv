# Note: we install the resulting java file in $prefix/bin since this is the only place where Windows searches dependent libraries
swigdir = $(bindir)
if COND_WIN32
  MAYBE_MINGW_CFLAGS = -Wl,--kill-at
endif
swig_LTLIBRARIES = libopenmbvjava.la libopenmbvjavaloadJNI.la
libopenmbvjava_la_SOURCES = OpenMBV_swig_java.cc
libopenmbvjava_la_LDFLAGS = -module -shrext .jni -avoid-version $(MAYBE_MINGW_CFLAGS) # use jni as so/dll extension for the java native binary file
libopenmbvjava_la_CPPFLAGS = -I$(top_srcdir) $(JAVA_JNICFLAGS) $(HDF5SERIE_CFLAGS) $(MBXMLUTILSHELPER_CFLAGS) -Wstrict-aliasing=0
libopenmbvjava_la_LIBADD = $(LDFLAGS_NOUNDEF) ../../openmbvcppinterface/libopenmbvcppinterface.la $(HDF5SERIE_LIBS) $(MBXMLUTILSHELPER_LIBS)
EXTRA_libopenmbvjava_la_DEPENDENCIES = openmbv.jar

libopenmbvjavaloadJNI_la_SOURCES = openmbvjavaloadjni.c
libopenmbvjavaloadJNI_la_LIBADD = $(LDFLAGS_NOUNDEF)
libopenmbvjavaloadJNI_la_LDFLAGS = -module -shrext .jni -avoid-version $(MAYBE_MINGW_CFLAGS) # use jni as so/dll extension for the java native binary file
libopenmbvjavaloadJNI_la_CPPFLAGS = $(JAVA_JNICFLAGS)

BUILT_SOURCES = OpenMBV_swig_java.cc

CLEANFILES = OpenMBV_swig_java.cc openmbv.jar OpenMBV_swig_java.d javatest.class
clean-local: clean-local-javasource
.PHONY: clean-local-javasource
clean-local-javasource:
	rm -rf de
.DELETE_ON_ERROR: # swig -Werror requires this
OpenMBV_swig_java.cc: $(srcdir)/../OpenMBV.i $(top_srcdir)/openmbvcppinterface/*.h
	rm -rf de/mbsim_env/openmbv
	$(MKDIR_P) de/mbsim_env/openmbv
	rm -f OpenMBV_swig_java.cc
	$(swig) -java -MP -MD -package de.mbsim_env.openmbv -outdir de/mbsim_env/openmbv -Werror -c++ -I$(top_srcdir) $(FMATVEC_CFLAGS) -o OpenMBV_swig_java.cc $(srcdir)/../OpenMBV.i
-include $(builddir)/OpenMBV_swig_java.d

de/mbsim_env/openmbv/Object.class: OpenMBV_swig_java.cc
	@JAVAC@ de/mbsim_env/openmbv/*.java
openmbv.jar: de/mbsim_env/openmbv/Object.class
	jar cf openmbv.jar de/mbsim_env/openmbv/*.class

install-exec-hook: openmbv.jar
	$(MKDIR_P) $(swigdir)
	$(INSTALL_DATA) openmbv.jar $(swigdir)/openmbv.jar
uninstall-local:
	rm -f $(swigdir)/openmbv.jar



TEST_EXTENSIONS=.sh
if COND_CROSSCOMPILE
else
  TESTS = javatest.sh
endif
check-local: javatest.class
javatest.class: $(srcdir)/javatest.java OpenMBV_swig_java.cc
	@JAVAC@ -cp . -d . $(srcdir)/javatest.java

exampledir = $(prefix)/share/openmbvcppinterface/examples/swig
example_DATA = $(srcdir)/javatest.java javatest.class $(srcdir)/matlabtest.m
