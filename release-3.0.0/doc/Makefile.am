all: openmbv/openmbv.tex

../schema/schema/openmbv.xsd: ../schema/schema/openmbv.xsd.in
	$(top_srcdir)/config.status --file=../schema/schema/openmbv.xsd

xml/index.xml: ../schema/schema/openmbv.xsd ../openmbvcppinterface-src/openmbvcppinterface/*.h
	doxygen

openmbv/openmbv.tex: xml/index.xml
	$$(pkg-config mbxmlutils --variable BINDIR)/mbxmlutils-xsd2html ../schema/schema/openmbv.xsd xml ../openmbvcppinterface-src/openmbvcppinterface OpenMBV OpenMBV:: true


install:
	cp -r http___* $$(pkg-config mbxmlutils --variable DOCDIR)



CLEANFILES=xml/index.xml openmbv/openmbv.tex
