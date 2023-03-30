SUBDIRS = openmbvcppinterface schema swig
DIST_SUBDIRS = openmbvcppinterface schema doc

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = openmbvcppinterface.pc

licdir = @datadir@/openmbvcppinterface
lic_DATA = COPYING NOTICE

# a hack for auto configure recreating on system updates, see configure.ac
$(CONFIG_STATUS_DEPENDENCIES):



# doxygen
.PHONY: doc doc_install doc_uninstall
doc:
	(cd doc && $(MAKE) $(AM_MAKEFLAGS) all)
doc_install:
	(cd doc && $(MAKE) $(AM_MAKEFLAGS) install)
doc_uninstall:
	(cd doc && $(MAKE) $(AM_MAKEFLAGS) uninstall)
