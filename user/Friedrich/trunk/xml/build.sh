#! /bin/sh

echo
echo
echo "RUN BY CONFIGURE"
echo

echo "Validate measurement.xml"
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /validate measurement.xml || exit
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

FILE=test.xml
NAMESPACE=http://www.amm.mw.tu-muenchen.de/YYY
NAMESPACELOCATION=test.xsd

BASENAME=$(basename $FILE .xml)

echo "Validate parameter.xml"
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /validate parameter.xml || exit
echo DONE

####### TODO: test for root element
echo "Validate $FILE"
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /validate $FILE /schema test.xsd || exit
echo DONE

#################################
###### TODO: validate all embeded files or/and validate output file !!!!!!!!!!!!
echo "EMBED"
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 embed.xsl /in $FILE /out .$BASENAME.embeded.xml || exit
echo DONE
#################################

echo "Rewrite all physical variables as octave-string in parameter.xml using physicalvariable2octavestring.xsl generating .parameter.octavestring.xml"
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 physicalvariable2octavestring.xsl /in parameter.xml /out .parameter.octavestring.xml || exit
echo DONE

echo "Rewrite all physical variables as octave-string in .$BASENAME.embeded.xml using physicalvariable2octavestring.xsl generating .$BASENAME.octavestring.xml"
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 physicalvariable2octavestring.xsl /in .$BASENAME.embeded.xml /out .$BASENAME.octavestring.xml || exit
echo DONE

echo "Resubstitute parameter using resubstitute.xsl generating .$BASENAME.resubstitute.xml (first generate resubstitute.xsl with proper namespace)"
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 setNamespace.xsl /in resubstitute.xsl /param "NAMESPACE='$NAMESPACE'" /param "NAMESPACELOCATION='$NAMESPACELOCATION'" /out .resubstitute.test.xsl || exit
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 .resubstitute.test.xsl /in .$BASENAME.octavestring.xml /out .$BASENAME.resubstitute.xml || exit
echo DONE

echo "Convert physical variables to SI units using convert2SIunit.xsl generating .$BASENAME.siunit.xml (first generate convert2SIunit.xsl with proper namespace)"
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 setNamespace.xsl /in convert2SIunit.xsl /param "NAMESPACE='$NAMESPACE'" /param "NAMESPACELOCATION='$NAMESPACELOCATION'" /out .convert2SIunit.test.xsl || exit
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 .convert2SIunit.test.xsl /in .$BASENAME.resubstitute.xml /out .$BASENAME.siunit.xml || exit
echo DONE

echo "Label physical variable expression for evaluation by octave using labelPhysicalVariable.xsl generating .$BASENAME.label.xml (first generate labelPhysicalVariable.xsl with proper namespace)"
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 setNamespace.xsl /in labelPhysicalVariable.xsl /param "NAMESPACE='$NAMESPACE'" /param "NAMESPACELOCATION='$NAMESPACELOCATION'" /out .labelPhysicalVariable.test.xsl || exit
wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2008/AltovaXML.exe /xslt2 .labelPhysicalVariable.test.xsl /in .$BASENAME.siunit.xml /out .$BASENAME.label.xml || exit
echo DONE

echo "Evaluate physical variable expression by octave using evaluatePhysicalVariable.m generating .$BASENAME.eval.xml"
octave -q evaluatePhysicalVariable.m .$BASENAME.label.xml > .$BASENAME.eval.xml || exit
echo DONE

echo "Output result"
cat .$BASENAME.eval.xml
#echo DONE
