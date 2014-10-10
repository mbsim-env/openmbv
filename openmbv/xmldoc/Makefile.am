all: openmbv/openmbv.tex

xml/index.xml: Doxyfile @MBXMLUTILSSCHEMADIR@/http___openmbv_berlios_de_OpenMBV/openmbv.xsd @OPENMBVCPPINTERFACEINCDIR@/openmbvcppinterface/*.h
	doxygen

openmbv/openmbv.tex: xml/index.xml
	@MBXMLUTILSBINDIR@/mbxmlutils-xsd2html @MBXMLUTILSSCHEMADIR@/http___openmbv_berlios_de_OpenMBV/openmbv.xsd xml @OPENMBVCPPINTERFACEINCDIR@/openmbvcppinterface OpenMBV OpenMBV::


install:
	cp -r http___* @MBXMLUTILSDOCDIR@



CLEANFILES=xml/index.xml openmbv/openmbv.tex
