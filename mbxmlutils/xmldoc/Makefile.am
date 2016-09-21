docdir = @datadir@/@PACKAGE@/doc
noinst_DATA = http___www_mbsim-env_de_MBXMLUtils/index.html

# generate measurement documentation
http___www_mbsim-env_de_MBXMLUtils/index.html: ../mbxmlutils/measurement.xml measurementToHtml.xsl ../mbxmlutils/.measurement.xml.isvalid ../mbxmlutils/physicalvariable.xsd
	DIR=$$(sed -rne '/ targetNamespace *=/s|^.*targetNamespace *= *"([^"]+)".*$$|\1|p' ../mbxmlutils/physicalvariable.xsd | sed -re 's|\.|_|g;s|:|_|g;s|/|_|g') || exit 1; \
	test -d $$DIR || mkdir $$DIR || exit 1; \
	@XALAN@ -o $$DIR/index.html -p DATETIME "'$$(date "+%a %b %d %Y %H:%M:%S")'" $(srcdir)/../mbxmlutils/measurement.xml $(srcdir)/measurementToHtml.xsl || exit 1; \
	sed -re "s/ xmlns:[^ ]+ *= *\"[^\"]+\"//g" $$DIR/index.html > $$DIR/index.html_ || exit 1; \
	mv -f $$DIR/index.html_ $$DIR/index.html || exit 1
# above line: Remove all namespace declaration from the html output using sed (maybe a bug in Xalan)

install-data-local:
	mkdir -p $(docdir)
	cp -rf http___* $(docdir)
