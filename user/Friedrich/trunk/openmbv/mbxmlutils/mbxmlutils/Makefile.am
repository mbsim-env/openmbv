# generate measurment docu
docdir = @datadir@/@PACKAGE@/doc
dist_doc_DATA = measurement.html

measurement.html: measurement.xml measurementToHtml.xsl
	@WINEALTOVAXML@ /validate measurement.xml
	@WINEALTOVAXML@ /xslt2 measurementToHtml.xsl /in measurement.xml /out measurement.html



# Generate physicalvariable.xsd
schemadir = @datadir@/@PACKAGE@/schema
dist_schema_DATA = physicalvariable.xsd

physicalvariable.xsd: measurement.xml measurement2physicalvariable.xsl
	@WINEALTOVAXML@ /validate measurement.xml
	@WINEALTOVAXML@ /xslt2 measurement2physicalvariable.xsl /in measurement.xml /out physicalvariable.xsd



CLEANFILES = measurement.html
