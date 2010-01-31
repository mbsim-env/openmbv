all: openmbv/openmbv.tex

xml/index.xml: ../schema/schema/openmbv.xsd ../openmbvcppinterface-src/openmbvcppinterface/*.h
	doxygen

openmbv/openmbv.tex: xml/index.xml
	$$(pkg-config mbxmlutils --variable BINDIR)/mbxmlutils-xsd2html $$(pkg-config mbxmlutils --variable SCHEMADIR)/http___openmbv_berlios_de_OpenMBV/openmbv.xsd xml ../openmbvcppinterface-src/openmbvcppinterface OpenMBV OpenMBV:: true


install:
	cp -r http___* $$(pkg-config mbxmlutils --variable DOCDIR)



CLEANFILES=xml/index.xml openmbv/openmbv.tex
