docdir = @datadir@/@PACKAGE@/doc
noinst_DATA = http___www_mbsim-env_de_MBXMLUtils/index.html

# generate measurement documentation
http___www_mbsim-env_de_MBXMLUtils/index.html: ../mbxmlutils/measurement.xml measurementToHtml.xsl ../mbxmlutils/.measurement.xml.isvalid ../mbxmlutils/physicalvariable.xsd
	DIR=$$(sed -rne '/ targetNamespace *=/s|^.*targetNamespace *= *"([^"]+)".*$$|\1|p' ../mbxmlutils/physicalvariable.xsd | sed -re 's|\.|_|g;s|:|_|g;s|/|_|g') || exit 1; \
	test -d $$DIR || mkdir $$DIR || exit 1; \
	@XALAN@ -o $$DIR/index.html -p DATETIME "'$$(date "+%a %b %d %Y %H:%M:%S")'" $(srcdir)/../mbxmlutils/measurement.xml $(srcdir)/measurementToHtml.xsl || exit 1; \
	sed -i -re "s/ xmlns:[^ ]+ *= *\"[^\"]+\"//g" $$DIR/index.html || exit 1
	RET=0; \
	$(prefix)/share/fmatvec/checkHtml/checkHtml.sh http___www_mbsim-env_de_MBXMLUtils/index.html || RET=1; \
	$(prefix)/share/fmatvec/checkHtml/checkMathJax.sh http___www_mbsim-env_de_MBXMLUtils/index.html || RET=1; \
	test $$RET -ne 0 && exit 1 || exit 0

install-data-local:
	mkdir -p $(docdir)
	cp -rf http___* $(docdir)

clean-local:
	rm -f http___www_mbsim-env_de_MBXMLUtils/index.html
