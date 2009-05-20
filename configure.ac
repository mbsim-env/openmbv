dnl Process this file with autoconf to produce a configure script.
AC_PREREQ([2.57])
AC_INIT([mbxmlutils],[1.0.0],[mafriedrich@users.berlios.de])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([xsd2html/addDoxygenToXsd.xsl])

AC_SUBST([DATADIR], ["$(eval echo $(eval echo $datadir))"])

dnl Check for xslt-2.0 (currently only wine + AltovaXML with fixed path)
WINEALTOVAXML="wine ~/.wine/drive_c/Program\ Files/Altova/AltovaXML2009/AltovaXML.exe"
eval $WINEALTOVAXML >& /dev/null
if test $? -ne 1; then
  AC_MSG_WARN([No xslt 2.0 found; Command failed: $WINEALTOVAXML]);
  WINEALTOVAXML="wine z:/home/OpenMBV/AltovaXML/AltovaXML.exe"
  eval $WINEALTOVAXML >& /dev/null
  if test $? -ne 1; then
    AC_MSG_ERROR([No xslt 2.0 found; Command failed: $WINEALTOVAXML]);
  fi
fi
AC_SUBST([WINEALTOVAXML])
DATADIR="$(eval echo $(eval echo $datadir))"
AC_SUBST([SCHEMADIR],[$DATADIR/$PACKAGE/schema])
AC_SUBST([XSLDIR],[$DATADIR/$PACKAGE/xsl])
AC_SUBST([OCTAVEDIR],[$DATADIR/$PACKAGE/octave])
AC_SUBST([XMLDIR],[z:$DATADIR/$PACKAGE/xml]) ### WINE (delete z: if not using wine)

AC_CONFIG_FILES([Makefile xsd2html/Makefile mbxmlutils/Makefile mbxmlutils/measurement.xml mbxmlutils.pc])
AC_CONFIG_FILES([xsd2html/mbxmlutils-xsd2html],[chmod +x xsd2html/mbxmlutils-xsd2html])
AC_CONFIG_FILES([mbxmlutils/mbxmlutils-simplifyxml],[chmod +x mbxmlutils/mbxmlutils-simplifyxml])

AC_OUTPUT
