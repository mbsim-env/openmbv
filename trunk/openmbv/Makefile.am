SUBDIRS = openmbv

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = openmbv.pc



# doc
.PHONY: doc doc_install doc_uninstall
doc:
	cd doc; \
	$(MAKE) $(AM_MAKEFLAGS)
doc_install:
	cd doc; \
	$(MAKE) $(AM_MAKEFLAGS) install
doc_uninstall:
	cd doc; \
	$(MAKE) $(AM_MAKEFLAGS) uninstall
