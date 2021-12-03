SUBDIRS = openmbv

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = openmbv.pc

licdir = @datadir@/openmbv
lic_DATA = COPYING NOTICE



# doc
.PHONY: doc doc_install doc_uninstall
doc:
	cd xmldoc; \
	$(MAKE) $(AM_MAKEFLAGS)
doc_install:
	cd xmldoc; \
	$(MAKE) $(AM_MAKEFLAGS) install
doc_uninstall:
	cd xmldoc; \
	$(MAKE) $(AM_MAKEFLAGS) uninstall
