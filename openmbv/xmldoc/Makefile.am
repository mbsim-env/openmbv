all: .openmbv.done

xml/index.xml: Doxyfile @MBXMLUTILSSCHEMADIR@/http___www_mbsim-env_de_OpenMBV/openmbv.xsd @OPENMBVCPPINTERFACEINCDIR@/openmbvcppinterface/*.h
	doxygen

.openmbv.done: xml/index.xml
	@MBXMLUTILSBINDIR@/mbxmlutils-xsd2html.sh @MBXMLUTILSSCHEMADIR@/http___www_mbsim-env_de_OpenMBV/openmbv.xsd xml @OPENMBVCPPINTERFACEINCDIR@/openmbvcppinterface OpenMBV OpenMBV::


install-data-local: .openmbv.done
	cp -r http___* @MBXMLUTILSDOCDIR@



CLEANFILES=xml/index.xml .openmbv.done
