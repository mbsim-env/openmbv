#! /bin/sh

echo
echo
echo "RUN BY CONFIGURE"
echo

FILE="/home/mbsim/local/share/mbsimXML/schema/mbsim.xsd"
PROJECT="MBSim"
DOXYGENDOC="/mnt/crypt/AMM/project/mbsim/kernel/doc/xml/all.xml"
DOXYGENCLASSPREFIX="MBSim::"
DOXYGENFUNCTIONPREFIX1="set"
DOXYGENFUNCTIONPREFIX2="get"

BASENAME=$(basename $FILE .xsd)

echo "Add Doxygen class and element-function documentation to xsd file $FILE using addDoxygenToXsdClass.xsl generating .$BASENAME.doxygen.xsd"
ln -sf $FILE .test.xml # AltavoXML.exe can not handle .xsd file as /in parameter !!!!!!
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 addDoxygenToXsdClass.xsl /in .test.xml \
  /param DOXYGENDOC="'$DOXYGENDOC'" \
  /param DOXYGENCLASSPREFIX="'$DOXYGENCLASSPREFIX'" \
  /param DOXYGENFUNCTIONPREFIX1="'$DOXYGENFUNCTIONPREFIX1'" \
  /param DOXYGENFUNCTIONPREFIX2="'$DOXYGENFUNCTIONPREFIX2'" \
  /out .$BASENAME.doxygen.xsd || exit
echo DONE

# CHECK USEDIN2 template
# CHECK inbetween ##########
# CHECK PHYSICALVARIABLEHTMLDOC ##########
echo "Generate HTML Docu from .$BASENAME.doxygen.xsd using xsdClassToHtml.xsl generating .$BASENAME.html"
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 xsdClassToHtml.xsl /in .$BASENAME.doxygen.xsd \
  /param PROJECT="'$PROJECT'" \
  /param PHYSICALVARIABLEHTMLDOC="'.measurement.html'" \
  /out .$BASENAME.html || exit
echo DONE

echo "Generate HTML Docu from ../measurement.xml using measurementToHtml.xsl generating .measurement.html"
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 measurementToHtml.xsl /in ../measurement.xml /out .measurement.html || exit
echo DONE

echo "Output result"
cat .measurement.html
