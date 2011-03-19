SUBDIRS = octave check

# distribute and install
schemadir = @datadir@/@PACKAGE@/schema/http___openmbv_berlios_de_MBXMLUtils
docdir = @datadir@/@PACKAGE@/doc
xmldir = @datadir@/@PACKAGE@/xml
dist_schema_DATA = physicalvariable.xsd parameter.xsd xml.xsd xhtml1-transitional.xsd
noinst_DATA = physicalvariable/physicalvariable.tex
dist_xml_DATA = measurement.xml





# validate measurement.xml
.measurement.xml.isvalid: measurement.xml measurement.xsd
	rm -f .measurement.xml.isvalid
	@XMLLINT@ --noout --schema measurement.xsd measurement.xml && touch .measurement.xml.isvalid



# generate measurement documentation
physicalvariable/physicalvariable.tex: measurement.xml measurementToTex.xsl measurementToHtml.xsl .measurement.xml.isvalid physicalvariable.xsd
	DIR=$$(sed -rne '/ targetNamespace *=/s|^.*targetNamespace *= *"([^"]+)".*$$|\1|p' physicalvariable.xsd | sed -re 's|\.|_|g;s|:|_|g;s|/|_|g') || exit 1; \
	@XSLTPROC@ --output $$DIR/index.xhtml --param DATETIME "'$$(date "+%a %b %d %Y %H:%M:%S")'" --param MBXMLUTILSVERSION "'@VERSION@'" measurementToHtml.xsl measurement.xml || exit 1; \
	@XMLLINT@ --noout --schema xhtml1-transitional.xsd $$DIR/index.xhtml || (rm $$DIR/index.xhtml; exit 1)
	@XSLTPROC@ --output physicalvariable/physicalvariable.tex measurementToTex.xsl measurement.xml
	cd physicalvariable || exit 1; \
	latex -help >& /dev/null && latex -interaction errorstopmode physicalvariable.tex || echo "Skip running LaTeX" || exit 1; \
	latex -help >& /dev/null && latex physicalvariable.tex >& /dev/null || echo "Skip running LaTeX" || exit 1; \
	latex -help >& /dev/null && latex physicalvariable.tex >& /dev/null || echo "Skip running LaTeX" || exit 1; \
	latex -help >& /dev/null && dvips -o physicalvariable.ps physicalvariable.dvi || echo "Skip running LaTeX"

install-data-local:
	mkdir -p $(docdir)
	cp -rf http___* $(docdir)
	



# generate physicalvariable.xsd
physicalvariable.xsd: measurement.xml measurement2physicalvariable.xsl .measurement.xml.isvalid
	@XSLTPROC@ --output physicalvariable.xsd measurement2physicalvariable.xsl measurement.xml
# link some files required for the "check" target to the "check" directory
	test -d check/http___openmbv_berlios_de_MBXMLUtils || mkdir check/http___openmbv_berlios_de_MBXMLUtils
	(cd check/http___openmbv_berlios_de_MBXMLUtils; \
	ln -fs ../../physicalvariable.xsd .; \
	ln -fs ../../xml.xsd .; \
	ln -fs ../../parameter.xsd .)



CLEANFILES = .measurement.xml.isvalid index.xhtml measurement/measurement.tex physicalvariable.xsd



# program to convert a complex xml file to a plain xml file
bin_PROGRAMS = mbxmlutilspp
mbxmlutilspp_SOURCES = preprocessor.cc
mbxmlutilspp_CPPFLAGS = $(LIBXML2_CFLAGS) -DTIXML_USE_STL $(MKOCTFILE_CFLAGS)
mbxmlutilspp_LDFLAGS = -L../mbxmlutilstinyxml
mbxmlutilspp_LDADD = $(LIBXML2_LIBS) -lmbxmlutilstinyxml $(MKOCTFILE_LIBS)
