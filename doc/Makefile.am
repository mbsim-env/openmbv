docdir = @datadir@/@PACKAGE@/doc
dist_doc_DATA = openmbv.html

../schema/openmbv.xsd: ../schema/openmbv.xsd.in
	$(top_srcdir)/config.status --file=../schema/openmbv.xsd

openmbv.html: ../schema/openmbv.xsd ../openmbvcppinterface/openmbvcppinterface/*.h
	doxygen
	$$(pkg-config mbxmlutils --variable BINDIR)/mbxmlutils-xsd2html ../schema/openmbv.xsd xml OpenMBV OpenMBV::
	sed -re "s|$$(pkg-config mbxmlutils --variable DOCDIR)/measurement.html|qrc:measurement.html|" ../doc/openmbv.html > openmbv.qt.html



schemadir = @datadir@/@PACKAGE@/schema
dist_schema_DATA = openmbv.doxygen.xsd

openmbv.doxygen.xsd: openmbv.html
