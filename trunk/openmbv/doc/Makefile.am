docdir = @datadir@/@PACKAGE@/doc
nodist_doc_DATA = openmbv.xhtml *.png

../schema/schema/openmbv.xsd: ../schema/schema/openmbv.xsd.in
	$(top_srcdir)/config.status --file=../schema/schema/openmbv.xsd

openmbv.xhtml: ../schema/schema/openmbv.xsd ../openmbvcppinterface-src/openmbvcppinterface/*.h
	doxygen
	$$(pkg-config mbxmlutils --variable BINDIR)/mbxmlutils-xsd2html ../schema/schema/openmbv.xsd xml .:../openmbvcppinterface-src/openmbvcppinterface OpenMBV OpenMBV:: true
	sed -re "s|$$(pkg-config mbxmlutils --variable DOCDIR)/measurement.xhtml|qrc:measurement.xhtml|;s|(<img class=\"[^\"]*\" src=\")([^\"]+)\"|\1qrc:\2\"|g" ../doc/openmbv.xhtml > openmbv.qt.xhtml



schemadir = @datadir@/@PACKAGE@/schema
nodist_schema_DATA = openmbv.doxygen.xsd

openmbv.doxygen.xsd: openmbv.xhtml

CLEANFILES=openmbv.xhtml
