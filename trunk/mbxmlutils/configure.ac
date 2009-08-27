dnl Process this file with autoconf to produce a configure script.
AC_PREREQ([2.57])
AC_INIT([mbxmlutils],[1.0.0],[mafriedrich@users.berlios.de])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([xsd2html/addDoxygenToXsd.xsl])
AC_CONFIG_HEADERS([config.h])
AC_PROG_LIBTOOL
AC_PROG_CC
AC_PROG_CXX
AC_LANG([C++])

dnl Check for MinGW
AC_CANONICAL_HOST
if test "_$host_os" = "_mingw32" ; then
  AC_DEFINE([MINGW], [1], [Host System is MinGW/MSys])
  LIBS="$LIBS -lregex"
fi

AC_SUBST([DATADIR], ["$(eval echo $(eval echo $datadir))"])

dnl Check for xslt
AC_ARG_WITH([xsltprocpath],
  [  --with-xsltproc-path=PATHTOXSLTPROC  The path to the 'xsltproc' program.],
  [xsltprocpath="$withval"],[xsltprocpath=""])
AC_PATH_PROG([xsltproc],[xsltproc],[AC_MSG_ERROR([xsltproc not found])],[$xsltprocpath:$prefix/bin:$PATH])
AC_SUBST([XSLTPROC],[$xsltproc])

dnl Check for libxml2
PKG_CHECK_MODULES(LIBXML2, libxml-2.0,[],[AC_MSG_ERROR([ERROR libxml2 not found])])

dnl Check for xmllint
AC_ARG_WITH([xmllintpath],
  [  --with-xmllint-path=PATHTOXMLLINT  The path to the 'xmllint' program.],
  [xmllintpath="$withval"],[xmllintpath=""])
AC_PATH_PROG([xmllint],[xmllint],[AC_MSG_ERROR([xmllint not found])],[$xmllintpath:$prefix/bin:$PATH])
AC_SUBST([XMLLINT],[$xmllint])

dnl Check for octave
AC_ARG_WITH([octave-path],
  [  --with-octave-path=PATHTOOCTAVE  The path to the 'octave' program.],
  [octavepath="$withval"],[octavepath=""])
AC_PATH_PROG([octave],[octave],[AC_MSG_ERROR([octave not found])],[$octavepath:$prefix/bin:$PATH])
AC_SUBST([OCTAVE],[$octave])
if [ !(echo "regexp('a','b'); regexprep('a','b','c');" | $octave -q) ]; then
  AC_MSG_ERROR([octave regexp and/or regexprep function not found! octave >= 3 needed!])
fi

dnl Helper functions to convert path to windows
dirtowin() {
  echo $(cd $1 && pwd -W 2> /dev/null)
}
filetowin() {
  echo $(cd $(dirname $1) && pwd -W 2> /dev/null)/$(basename $1)
}

DATADIR="$(eval echo $(eval echo $datadir))"
AC_SUBST([SCHEMADIR],[$DATADIR/$PACKAGE/schema])
AC_SUBST([XSLDIR],[$DATADIR/$PACKAGE/xsl])
AC_SUBST([OCTAVEDIR],[$DATADIR/$PACKAGE/octave])
AC_SUBST([XMLDIR],[$DATADIR/$PACKAGE/xml])
if test "_$host_os" = "_mingw32" ; then
  mkdir -p $SCHEMADIR $XMLDIR $OCTAVEDIR
  SCHEMADIR=$(dirtowin $SCHEMADIR)
  XMLDIR=$(dirtowin $XMLDIR)
  OCTAVEDIR=$(dirtowin $OCTAVEDIR)
  OCTAVE=$(filetowin $OCTAVE)
fi

AC_CONFIG_FILES([Makefile xsd2html/Makefile mbxmlutilstinyxml/Makefile mbxmlutils/Makefile mbxmlutils/measurement.xml mbxmlutils.pc mbxmlutils/env.h xsd2html/xhtml1-transitional.xsd])
AC_CONFIG_FILES([xsd2html/mbxmlutils-xsd2html],[chmod +x xsd2html/mbxmlutils-xsd2html])

AC_OUTPUT
