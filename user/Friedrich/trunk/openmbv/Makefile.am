if COND_MBXMLUTILS
  MAYBE_SCHEMA = schema
endif

SUBDIRS = tinyxml openmbv $(MAYBE_SCHEMA)



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
