# Note: we install the resulting java file in $prefix/bin since this is the only place where Windows searches dependent libraries
swigdir = $(bindir)
if COND_MINGW
  MAYBE_MINGW_CFLAGS = -Wl,--kill-at
endif
swig_LTLIBRARIES = libopenmbvjava.la
libopenmbvjava_la_SOURCES = OpenMBV_swig_java.cc
libopenmbvjava_la_LDFLAGS = -module -Wl,-rpath,\$$ORIGIN/../lib -shrext .jni -avoid-version $(MAYBE_MINGW_CFLAGS) # use jni as so/dll extension for the java native binary file
libopenmbvjava_la_CPPFLAGS = $(MBXMLUTILSTINYXML_CFLAGS) $(JAVA_JNICFLAGS) $(HDF5SERIE_CFLAGS) -Wno-error=strict-aliasing -Wno-strict-aliasing
libopenmbvjava_la_LIBADD = ../../openmbvcppinterface/libopenmbvcppinterface.la
EXTRA_libopenmbvjava_la_DEPENDENCIES = openmbv.jar

CLEANFILES=OpenMBV_swig_java.cc openmbv.jar
clean-local: clean-local-javasource
.PHONY: clean-local-javasource
clean-local-javasource:
	rm -rf de
OpenMBV_swig_java.cc: ../OpenMBV.i
	$(MKDIR_P) de/berlios/openmbv/OpenMBV
	$(swig) -java -package de.berlios.openmbv.OpenMBV -outdir de/berlios/openmbv/OpenMBV -Werror -c++ -I../.. -o OpenMBV_swig_java.cc ../OpenMBV.i

de/berlios/openmbv/OpenMBV/Object.class: OpenMBV_swig_java.cc
	@JAVAC@ de/berlios/openmbv/OpenMBV/*.java
openmbv.jar: de/berlios/openmbv/OpenMBV/Object.class javatest.class # this is a hack: javatest.class should be build only for check target
	jar cf openmbv.jar de/berlios/openmbv/OpenMBV/*.class

install-exec-hook: openmbv.jar
	$(MKDIR_P) $(swigdir)
	$(INSTALL_DATA) openmbv.jar $(swigdir)/openmbv.jar
uninstall-local:
	rm -f $(swigdir)/openmbv.jar



TESTS = javatest.sh
javatest.class: javatest.java
	@JAVAC@ javatest.java

exampledir = $(prefix)/share/openmbvcppinterface/examples/swig
example_DATA = javatest.java matlabtest.m
