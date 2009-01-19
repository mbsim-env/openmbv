#! /bin/sh

echo
echo
echo "RUN BY CONFIGURE"
echo

echo "Validate measurement.xml"
/home/mbsim/local/bin/xmlutils-parse measurement.xml || exit
echo DONE

echo "Generate physicalvariable.xsd by measurement.xml using measurement2physicalvariable.xsl"
/home/mbsim/local/bin/xmlutils-xslt measurement2physicalvariable.xsl measurement.xml > physicalvariable.xsd || exit
echo DONE

echo "Generate convert2SIunit.xsl by measurement.xml using measurement2convert2SIunit.xsl"
/home/mbsim/local/bin/xmlutils-xslt measurement2convert2SIunit.xsl measurement.xml > convert2SIunit.xsl || exit
echo DONE



echo
echo
echo "RUN BY MBSIM"
echo
echo "run with param: test.xml http://www.amm.mw.tu-muenchen.de/YYY test.xsd"
echo

FILE=test.xml
NAMESPACE=http://www.amm.mw.tu-muenchen.de/YYY
NAMESPACELOCATION=test.xsd

DIR=$(dirname $FILE)
BASENAME=$(basename $FILE .xml)

echo "Validate parameter.xml"
/home/mbsim/local/bin/xmlutils-parse parameter.xml || exit
echo DONE

####### TODO: test for root element
echo "Validate $FILE"
/home/mbsim/local/bin/xmlutils-parse $FILE || exit
echo DONE

####### TODO: export files to check from $FILE; test for root element in files
echo "Validate testm.xml, testv.xml"
/home/mbsim/local/bin/xmlutils-parse testm.xml || exit
/home/mbsim/local/bin/xmlutils-parse testv.xml || exit
echo DONE



echo "Rewrite all physical variables as octave-string in parameter.xml using physicalvariable2octavestring.xsl generating .parameter.octavestring.xml"
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 physicalvariable2octavestring.xsl /in parameter.xml /out .parameter.octavestring.xml || exit
echo DONE

echo "Rewrite all physical variables as octave-string in $FILE using physicalvariable2octavestring.xsl generating $DIR/.$BASENAME.octavestring.xml"
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 physicalvariable2octavestring.xsl /in $FILE /out $DIR/.$BASENAME.octavestring.xml || exit
echo DONE

echo "Resubstitute parameter using resubstitute.xsl generating $DIR/.$BASENAME.resubstitute.xml (first generate resubstitute.xsl with proper namespace)"
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 setNamespace.xsl /in resubstitute.xsl /param "NAMESPACE='$NAMESPACE'" /param "NAMESPACELOCATION='$NAMESPACELOCATION'" /out .resubstitute.test.xsl || exit
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 .resubstitute.test.xsl /in $DIR/.$BASENAME.octavestring.xml /out $DIR/.$BASENAME.resubstitute.xml || exit
echo DONE

# TODO: test only for units of specific measure and not for all units
echo "Convert physical variables to SI units using convert2SIunit.xsl generating $DIR/.$BASENAME.siunit.xml (first generate convert2SIunit.xsl with proper namespace)"
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 setNamespace.xsl /in convert2SIunit.xsl /param "NAMESPACE='$NAMESPACE'" /param "NAMESPACELOCATION='$NAMESPACELOCATION'" /out .convert2SIunit.test.xsl || exit
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 .convert2SIunit.test.xsl /in $DIR/.$BASENAME.resubstitute.xml /out $DIR/.$BASENAME.siunit.xml || exit
echo DONE

echo "Label physical variable expression for evaluation by octave using labelPhysicalVariable.xsl generating $DIR/.$BASENAME.label.xml (first generate labelPhysicalVariable.xsl with proper namespace)"
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 setNamespace.xsl /in labelPhysicalVariable.xsl /param "NAMESPACE='$NAMESPACE'" /param "NAMESPACELOCATION='$NAMESPACELOCATION'" /out .labelPhysicalVariable.test.xsl || exit
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 .labelPhysicalVariable.test.xsl /in $DIR/.$BASENAME.siunit.xml /out $DIR/.$BASENAME.label.xml || exit
echo DONE

echo "Evaluate physical variable expression by octave using evaluatePhysicalVariable.m generating $DIR/.$BASENAME.eval.xml"
octave -q evaluatePhysicalVariable.m $DIR/.$BASENAME.label.xml > $DIR/.$BASENAME.eval.xml || exit
echo DONE

echo "Output result"
cat $DIR/.$BASENAME.eval.xml
#echo DONE
