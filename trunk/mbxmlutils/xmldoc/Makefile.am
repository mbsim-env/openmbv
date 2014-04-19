docdir = @datadir@/@PACKAGE@/doc
noinst_DATA = physicalvariable/physicalvariable.tex

# generate measurement documentation
physicalvariable/physicalvariable.tex: ../mbxmlutils/measurement.xml measurementToTex.xsl measurementToHtml.xsl ../mbxmlutils/.measurement.xml.isvalid ../mbxmlutils/physicalvariable.xsd
	DIR=$$(sed -rne '/ targetNamespace *=/s|^.*targetNamespace *= *"([^"]+)".*$$|\1|p' ../mbxmlutils/physicalvariable.xsd | sed -re 's|\.|_|g;s|:|_|g;s|/|_|g') || exit 1; \
	@XALAN@ -o $$DIR/index.xhtml -p DATETIME "'$$(date "+%a %b %d %Y %H:%M:%S")'" -p MBXMLUTILSVERSION "'@VERSION@'" ../mbxmlutils/measurement.xml measurementToHtml.xsl || exit 1; \
	../mbxmlutilshelper/mbxmlutilsvalidate$(EXEEXT) ../mbxmlutilshelper/xhtml1-transitional.xsd $$DIR/index.xhtml || (rm $$DIR/index.xhtml; exit 1)
	@XALAN@ -o physicalvariable/physicalvariable.tex ../mbxmlutils/measurement.xml measurementToTex.xsl
	cd physicalvariable || exit 1; \
	latex -help >& /dev/null && latex -interaction errorstopmode physicalvariable.tex || echo "Skip running LaTeX" || exit 1; \
	latex -help >& /dev/null && latex physicalvariable.tex >& /dev/null || echo "Skip running LaTeX" || exit 1; \
	latex -help >& /dev/null && latex physicalvariable.tex >& /dev/null || echo "Skip running LaTeX" || exit 1; \
	latex -help >& /dev/null && dvips -o physicalvariable.ps physicalvariable.dvi || echo "Skip running LaTeX"

install-data-local:
	mkdir -p $(docdir)
	cp -rf http___* $(docdir)
