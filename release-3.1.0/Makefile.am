if COND_XML
  MAYBE_SCHEMA = schema
endif

SUBDIRS = openmbvcppinterfacetinyxml openmbvcppinterface $(MAYBE_SCHEMA)

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = openmbvcppinterface.pc



# doxygen
.PHONY: doc doc_install doc_uninstall
doc:
	cd doc; \
	rm -rf html; \
	$(doxygen)
doc_install:
	cd doc; \
	$(MKDIR_P) $(docdir); \
	cp -r html $(docdir); \
	cp $(PACKAGE).tag $(docdir)
doc_uninstall:
	rm -rf $(docdir)/html
	rm -rf $(docdir)/$(PACKAGE).tag
