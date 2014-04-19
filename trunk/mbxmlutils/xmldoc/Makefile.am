docdir = @datadir@/@PACKAGE@/doc
noinst_DATA = physicalvariable/physicalvariable.tex

# generate measurement documentation
physicalvariable/physicalvariable.tex: ../mbxmlutils/measurement.xml measurementToTex.xsl measurementToHtml.xsl ../mbxmlutils/.measurement.xml.isvalid ../mbxmlutils/physicalvariable.xsd
	DIR=$$(sed -rne '/ targetNamespace *=/s|^.*targetNamespace *= *"([^"]+)".*$$|\1|p' ../mbxmlutils/physicalvariable.xsd | sed -re 's|\.|_|g;s|:|_|g;s|/|_|g') || exit 1; \
	@XSLTPROC@ --output $$DIR/index.xhtml --param DATETIME "'$$(date "+%a %b %d %Y %H:%M:%S")'" --param MBXMLUTILSVERSION "'@VERSION@'" measurementToHtml.xsl ../mbxmlutils/measurement.xml || exit 1; \
	../mbxmlutilshelper/mbxmlutilsvalidate$(EXEEXT) ../mbxmlutilshelper/xhtml1-transitional.xsd $$DIR/index.xhtml || (rm $$DIR/index.xhtml; exit 1)
	@XSLTPROC@ --output physicalvariable/physicalvariable.tex measurementToTex.xsl ../mbxmlutils/measurement.xml
	cd physicalvariable || exit 1; \
	latex -help >& /dev/null && latex -interaction errorstopmode physicalvariable.tex || echo "Skip running LaTeX" || exit 1; \
	latex -help >& /dev/null && latex physicalvariable.tex >& /dev/null || echo "Skip running LaTeX" || exit 1; \
	latex -help >& /dev/null && latex physicalvariable.tex >& /dev/null || echo "Skip running LaTeX" || exit 1; \
	latex -help >& /dev/null && dvips -o physicalvariable.ps physicalvariable.dvi || echo "Skip running LaTeX"

install-data-local:
	mkdir -p $(docdir)
	cp -rf http___* $(docdir)
