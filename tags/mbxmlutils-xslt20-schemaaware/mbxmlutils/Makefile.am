# distribute and install
schemadir = @datadir@/@PACKAGE@/schema
docdir = @datadir@/@PACKAGE@/doc
xsldir = @datadir@/@PACKAGE@/xsl
octavedir = @datadir@/@PACKAGE@/octave
xmldir = @datadir@/@PACKAGE@/xml
dist_schema_DATA = physicalvariable.xsd parameter.xsd xml.xsd
dist_doc_DATA = measurement.html
dist_xsl_DATA = convert2SIunit.xsl embed.xsl setNamespace.xsl physicalvariable2octavenotation.xsl resubstitute.xsl labelPhysicalVariable.xsl
dist_octave_DATA = evaluatePhysicalVariable.m
dist_xml_DATA = measurement.xml



# validate measurement.xml
.measurement.xml.isvalid: measurement.xml measurement.xsd
	rm -f .measurement.xml.isvalid
	@WINEALTOVAXML@ /validate measurement.xml /schema measurement.xsd && touch .measurement.xml.isvalid



# generate measurement documentation
measurement.html: measurement.xml measurementToHtml.xsl .measurement.xml.isvalid
	@WINEALTOVAXML@ /xslt2 measurementToHtml.xsl /in measurement.xml /out measurement.html



# generate physicalvariable.xsd
physicalvariable.xsd: measurement.xml measurement2physicalvariable.xsl .measurement.xml.isvalid
	@WINEALTOVAXML@ /xslt2 measurement2physicalvariable.xsl /in measurement.xml /out physicalvariable.xsd



# generate convert2SIunit.xsl
convert2SIunit.xsl: measurement.xml measurement2convert2SIunit.xsl .measurement.xml.isvalid
	@WINEALTOVAXML@ /xslt2 measurement2convert2SIunit.xsl /in measurement.xml /param "XMLDIR='@XMLDIR@'" /out convert2SIunit.xsl



CLEANFILES = .measurement.xml.isvalid measurement.html physicalvariable.xsd convert2SIunit.xsl



# script to convert a complex xml file to a plain xml file
bin_SCRIPTS = mbxmlutils-simplifyxml
