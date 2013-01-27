if NOTONLY_MBXMLUTILSTINYXML
  ADDITIONAL_SUBDIRS = mbxmlutils xsd2html
  ADDITIONAL_PKG_CONFIG = mbxmlutils.pc
endif
SUBDIRS = mbxmlutilstinyxml $(ADDITIONAL_SUBDIRS)
DIST_SUBDIRS = mbxmlutilstinyxml $(ADDITIONAL_SUBDIRS) xmldoc

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = $(ADDITIONAL_PKG_CONFIG) mbxmlutilstinyxml.pc
