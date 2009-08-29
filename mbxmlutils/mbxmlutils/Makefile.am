# distribute and install
schemadir = @datadir@/@PACKAGE@/schema
docdir = @datadir@/@PACKAGE@/doc
xmldir = @datadir@/@PACKAGE@/xml
octavedir = @datadir@/@PACKAGE@/octave
dist_schema_DATA = physicalvariable.xsd parameter.xsd xml.xsd
dist_doc_DATA = measurement.xhtml measurement.ps
dist_xml_DATA = measurement.xml
dist_octave_DATA = evaluate.m



# validate measurement.xml
.measurement.xml.isvalid: measurement.xml measurement.xsd
	rm -f .measurement.xml.isvalid
	@XMLLINT@ --noout --schema measurement.xsd measurement.xml && touch .measurement.xml.isvalid



# generate measurement documentation
measurement.xhtml: measurement.xml measurementToHtml.xsl .measurement.xml.isvalid
	@XSLTPROC@ --output measurement.xhtml measurementToHtml.xsl measurement.xml
	@XMLLINT@ --noout --schema $(datadir)/$(PACKAGE)/schema/xhtml1-transitional.xsd measurement.xhtml || rm measurement.xhtml
measurement.ps: measurement.xml measurementToHtml.xsl .measurement.xml.isvalid
	@XSLTPROC@ --output measurement.tex measurementToTex.xsl measurement.xml
	latex -interaction errorstopmode measurement.tex
	latex measurement.tex >& /dev/null
	latex measurement.tex >& /dev/null
	dvips -o measurement.ps measurement.dvi



# generate physicalvariable.xsd
physicalvariable.xsd: measurement.xml measurement2physicalvariable.xsl .measurement.xml.isvalid
	@XSLTPROC@ --param SCHEMADIR "'@SCHEMADIR@'" --output physicalvariable.xsd measurement2physicalvariable.xsl measurement.xml



CLEANFILES = .measurement.xml.isvalid measurement.xhtml measurement.ps physicalvariable.xsd



# program to convert a complex xml file to a plain xml file
bin_PROGRAMS = mbxmlutilspp
mbxmlutilspp_SOURCES = preprocessor.cc
mbxmlutilspp_CPPFLAGS = $(LIBXML2_CFLAGS) -DTIXML_USE_STL
mbxmlutilspp_LDFLAGS = -L../mbxmlutilstinyxml
mbxmlutilspp_LDADD = $(LIBXML2_LIBS) -lmbxmlutilstinyxml
