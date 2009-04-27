docdir = @datadir@/@PACKAGE@/doc
dist_doc_DATA = openmbv.html

openmbv.html: ../schema/openmbv.xsd
	doxygen
	$$(pkg-config mbxmlutils --variable BINDIR)/mbxmlutils-xsd2html ../schema/openmbv.xsd xml OpenMBV OpenMBV::



schemadir = @datadir@/@PACKAGE@/schema
dist_schema_DATA = openmbv.doxygen.xsd

openmbv.doxygen.xsd: openmbv.html
