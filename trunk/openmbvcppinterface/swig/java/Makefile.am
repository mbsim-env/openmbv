# Note: we install the resulting java file in $prefix/bin since this is the only place where Windows searches dependent libraries
swigdir = $(bindir)
if COND_MINGW
  MAYBE_MINGW_CFLAGS = -Wl,--kill-at
endif
swig_LTLIBRARIES = libopenmbvjava.la libopenmbvjavaloadJNI.la
libopenmbvjava_la_SOURCES = OpenMBV_swig_java.cc
libopenmbvjava_la_LDFLAGS = -module -Wl,-rpath,\$$ORIGIN/../lib -shrext .jni -avoid-version $(MAYBE_MINGW_CFLAGS) # use jni as so/dll extension for the java native binary file
libopenmbvjava_la_CPPFLAGS = $(JAVA_JNICFLAGS) $(HDF5SERIE_CFLAGS) -Wno-error=strict-aliasing -Wno-strict-aliasing
libopenmbvjava_la_LIBADD = ../../openmbvcppinterface/libopenmbvcppinterface.la
EXTRA_libopenmbvjava_la_DEPENDENCIES = openmbv.jar

libopenmbvjavaloadJNI_la_SOURCES = openmbvjavaloadjni.c
libopenmbvjavaloadJNI_la_LDFLAGS = -module -Wl,-rpath,\$$ORIGIN/../lib -shrext .jni -avoid-version $(MAYBE_MINGW_CFLAGS) # use jni as so/dll extension for the java native binary file
libopenmbvjavaloadJNI_la_CPPFLAGS = $(JAVA_JNICFLAGS)

CLEANFILES=OpenMBV_swig_java.cc openmbv.jar
clean-local: clean-local-javasource
.PHONY: clean-local-javasource
clean-local-javasource:
	rm -rf de
OpenMBV_swig_java.cc: ../OpenMBV.i ../../openmbvcppinterface/*.h
	$(MKDIR_P) de/berlios/openmbv/OpenMBV
	$(swig) -java -package de.berlios.openmbv.OpenMBV -outdir de/berlios/openmbv/OpenMBV -Werror -c++ -I../.. $(FMATVEC_CFLAGS) -o OpenMBV_swig_java.cc ../OpenMBV.i || (rm -f OpenMBV_swig_java.cc; exit 1)

de/berlios/openmbv/OpenMBV/Object.class: OpenMBV_swig_java.cc
	@JAVAC@ de/berlios/openmbv/OpenMBV/*.java
openmbv.jar: de/berlios/openmbv/OpenMBV/Object.class
	jar cf openmbv.jar de/berlios/openmbv/OpenMBV/*.class

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
javatest.log: javatest.class
javatest.class: javatest.java
	@JAVAC@ javatest.java

exampledir = $(prefix)/share/openmbvcppinterface/examples/swig
example_DATA = javatest.java matlabtest.m
