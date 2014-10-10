docdir = @datadir@/@PACKAGE@/doc
noinst_DATA = http___openmbv_berlios_de_MBXMLUtils_physicalvariable/index.xhtml

# generate measurement documentation
http___openmbv_berlios_de_MBXMLUtils_physicalvariable/index.xhtml: ../mbxmlutils/measurement.xml measurementToHtml.xsl ../mbxmlutils/.measurement.xml.isvalid ../mbxmlutils/physicalvariable.xsd
	DIR=$$(sed -rne '/ targetNamespace *=/s|^.*targetNamespace *= *"([^"]+)".*$$|\1|p' ../mbxmlutils/physicalvariable.xsd | sed -re 's|\.|_|g;s|:|_|g;s|/|_|g') || exit 1; \
	test -d $$DIR || mkdir $$DIR || exit 1; \
	@XALAN@ -o $$DIR/index.xhtml -p DATETIME "'$$(date "+%a %b %d %Y %H:%M:%S")'" -p MBXMLUTILSVERSION "'@VERSION@'" $(srcdir)/../mbxmlutils/measurement.xml $(srcdir)/measurementToHtml.xsl || exit 1; \
	../mbxmlutilshelper/mbxmlutilsvalidate$(EXEEXT) $(srcdir)/../mbxmlutilshelper/xhtml1-transitional.xsd $$DIR/index.xhtml || (rm $$DIR/index.xhtml; exit 1)

install-data-local:
	mkdir -p $(docdir)
	cp -rf http___* $(docdir)
