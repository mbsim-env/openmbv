if COND_PYTHON
  MAYBE_PYTHON_DIR = python
endif

if COND_JAVA
  MAYBE_JAVA_DIR = java
endif

SUBDIRS = octave $(MAYBE_PYTHON_DIR) $(MAYBE_JAVA_DIR)

swigdir = $(datadir)/openmbvcppinterface
swig_DATA = OpenMBV.i OpenMBV_include.i
