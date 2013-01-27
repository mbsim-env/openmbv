SUBDIRS = openmbv

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = openmbv.pc



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
