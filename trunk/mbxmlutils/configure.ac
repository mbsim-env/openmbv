dnl Process this file with autoconf to produce a configure script.
AC_PREREQ([2.57])
AC_INIT([mbxmlutils],[1.1.0],[mafriedrich@users.berlios.de])
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
  # MINGW is already in use by octave header
  AC_DEFINE([MBXMLUTILS_MINGW], [1], [Host System is MinGW/MSys])
  CPPFLAGS="$CPPFLAGS -DMBXMLUTILS_MINGW"
  LIBS="$LIBS -lregex"
fi

LDFLAGS="-Wl,--no-undefined"
CXXFLAGS="-std=c++0x"

AC_SUBST([DATADIR], ["$(eval echo $(eval echo $datadir))"])

dnl Check for xslt
AC_ARG_WITH([xsltprocpath],
  [  --with-xsltproc-path=PATHTOXSLTPROC  The path to the 'xsltproc' program.],
  [xsltprocpath="$withval"],[xsltprocpath=""])
AC_PATH_PROG([xsltproc],[xsltproc],[no],[$xsltprocpath:$prefix/bin:$PATH])
test $xsltproc = "no" && AC_MSG_ERROR([xsltproc not found])
AC_SUBST([XSLTPROC],[$xsltproc])

dnl Check for libxml2
PKG_CHECK_MODULES(LIBXML2, libxml-2.0,[],[AC_MSG_ERROR([ERROR libxml2 not found])])

dnl Check for xmllint
AC_ARG_WITH([xmllintpath],
  [  --with-xmllint-path=PATHTOXMLLINT  The path to the 'xmllint' program.],
  [xmllintpath="$withval"],[xmllintpath=""])
AC_PATH_PROG([xmllint],[xmllint],[no],[$xmllintpath:$prefix/bin:$PATH])
test $xmllint = "no" && AC_MSG_ERROR([xmllint not found])
AC_SUBST([XMLLINT],[$xmllint])

dnl Check for octave
AC_ARG_WITH([mkoctfile-path],
  [  --with-mkoctfile-path=PATHTOMKOCTFILE  The path to the 'mkoctfile' program.],
  [mkoctfilepath="$withval"],[mkoctfilepath=""])
AC_PATH_PROG([mkoctfile],[mkoctfile],[no],[$mkoctfilepath:$prefix/bin:$PATH])
test $mkoctfile = "no" && AC_MSG_ERROR([mkoctfile not found])
AC_SUBST([MKOCTFILE],[$mkoctfile])
# use allways the same compiler as mkoctfile for CXX
AC_PATH_PROG([CXX],[$($MKOCTFILE -p CXX)],[no],[$mkoctfilepath:$mkoctfilepath/../mingw32/bin:$prefix/bin:$PATH])
test $CXX = "no" && AC_MSG_ERROR([mkoctfile compiler not found])
# get mkoctfile compiler flags
MKOCTFILE_CFLAGS="$(echo $($MKOCTFILE -p INCFLAGS) | sed -re 's|\\|/|g')"
MKOCTFILE_LIBS="$(echo $($MKOCTFILE -p RLD_FLAG) $($MKOCTFILE -p LFLAGS) $($MKOCTFILE -p OCTAVE_LIBS) | sed -re 's|\\|/|g')"
AC_SUBST([MKOCTFILE_CFLAGS])
AC_SUBST([MKOCTFILE_LIBS])
# try to compile a octave stand alone program
SAVE_CPPFLAGS=$CPPFLAGS; CPPFLAGS="$CPPFLAGS $MKOCTFILE_CFLAGS"
SAVE_LIBS=$LIBS; LIBS="$LIBS $MKOCTFILE_LIBS"
AC_CHECKING(compiling a octave program)
AC_LINK_IFELSE([AC_LANG_PROGRAM([[
#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>]],[[
  const char *octave_argv[2]={"dummy", "-q"};
  octave_main(2, (char**)octave_argv, 1);
  int dummy;
  eval_string("1+1;",true,dummy,0);
]])], AC_MSG_RESULT([done]), AC_MSG_ERROR([can not compile a octave program]))
CPPFLAGS=$SAVE_CPPFLAGS
LIBS=$SAVE_LIBS

# The prefix for octave m and oct file search path can be changed
# using the octave envvar OCTAVE_HOME.



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
fi

CPPFLAGS="$CPPFLAGS -Wall -Werror"

AC_CONFIG_FILES([Makefile xsd2html/Makefile mbxmlutilstinyxml/Makefile mbxmlutils/Makefile mbxmlutils.pc mbxmlutils/env.h mbxmlutils/octave/Makefile])
AC_CONFIG_FILES([xsd2html/mbxmlutils-xsd2html],[chmod +x xsd2html/mbxmlutils-xsd2html])

AC_OUTPUT
