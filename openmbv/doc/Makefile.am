docdir = @datadir@/@PACKAGE@/doc
nodist_doc_DATA = openmbv.html *.png

../schema/schema/openmbv.xsd: ../schema/schema/openmbv.xsd.in
	$(top_srcdir)/config.status --file=../schema/schema/openmbv.xsd

openmbv.html: ../schema/schema/openmbv.xsd ../openmbvcppinterface-src/openmbvcppinterface/*.h
	doxygen
	$$(pkg-config mbxmlutils --variable BINDIR)/mbxmlutils-xsd2html ../schema/schema/openmbv.xsd xml html OpenMBV OpenMBV::
	sed -re "s|$$(pkg-config mbxmlutils --variable DOCDIR)/measurement.html|qrc:measurement.html|" ../doc/openmbv.html > openmbv.qt.html



schemadir = @datadir@/@PACKAGE@/schema
nodist_schema_DATA = openmbv.doxygen.xsd

openmbv.doxygen.xsd: openmbv.html
