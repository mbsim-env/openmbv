if NOTONLY_MBXMLUTILSTINYXML
  ADDITIONAL_SUBDIRS = mbxmlutils xsd2html
  ADDITIONAL_PKG_CONFIG = mbxmlutils.pc
endif
SUBDIRS = mbxmlutilshelper mbxmlutilstinyxml $(ADDITIONAL_SUBDIRS)
DIST_SUBDIRS = mbxmlutilshelper mbxmlutilstinyxml $(ADDITIONAL_SUBDIRS) xmldoc doc

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = $(ADDITIONAL_PKG_CONFIG) mbxmlutilstinyxml.pc mbxmlutilshelper.pc



# doxygen
.PHONY: doc doc_install doc_uninstall
doc:
	(cd doc && $(MAKE) $(AM_MAKEFLAGS) all)
doc_install:
	(cd doc && $(MAKE) $(AM_MAKEFLAGS) install)
doc_uninstall:
	(cd doc && $(MAKE) $(AM_MAKEFLAGS) uninstall)
