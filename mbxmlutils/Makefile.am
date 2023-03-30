SUBDIRS = mbxmlutilshelper mbxmlutils xsd2html
DIST_SUBDIRS = mbxmlutilshelper mbxmlutils xsd2html xmldoc doc

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = mbxmlutils.pc mbxmlutilshelper.pc

licdir = @datadir@/mbxmlutils
lic_DATA = COPYING NOTICE



# doxygen
.PHONY: doc doc_install doc_uninstall
doc:
	(cd doc && $(MAKE) $(AM_MAKEFLAGS) all)
doc_install:
	(cd doc && $(MAKE) $(AM_MAKEFLAGS) install)
doc_uninstall:
	(cd doc && $(MAKE) $(AM_MAKEFLAGS) uninstall)
