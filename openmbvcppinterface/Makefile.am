if COND_XML
  MAYBE_SCHEMA = schema
endif

if COND_SWIG
  MAYBE_SWIG = swig
endif

SUBDIRS = openmbvcppinterface $(MAYBE_SCHEMA) $(MAYBE_SWIG)
DIST_SUBDIRS = openmbvcppinterface $(MAYBE_SCHEMA) doc

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = openmbvcppinterface.pc



# doxygen
.PHONY: doc doc_install doc_uninstall
doc:
	(cd doc && $(MAKE) $(AM_MAKEFLAGS) all)
doc_install:
	(cd doc && $(MAKE) $(AM_MAKEFLAGS) install)
doc_uninstall:
	(cd doc && $(MAKE) $(AM_MAKEFLAGS) uninstall)
