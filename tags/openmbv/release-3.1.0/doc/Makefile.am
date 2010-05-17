all: openmbv/openmbv.tex

xml/index.xml: ../schema/schema/openmbv.xsd ../openmbvcppinterface-src/openmbvcppinterface/*.h
	doxygen

openmbv/openmbv.tex: xml/index.xml
	@MBXMLUTILSBINDIR@/mbxmlutils-xsd2html @MBXMLUTILSSCHEMADIR@/http___openmbv_berlios_de_OpenMBV/openmbv.xsd xml ../openmbvcppinterface-src/openmbvcppinterface OpenMBV OpenMBV:: true


install:
	cp -r http___* @MBXMLUTILSDOCDIR@



CLEANFILES=xml/index.xml openmbv/openmbv.tex
