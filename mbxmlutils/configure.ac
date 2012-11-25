dnl Process this file with autoconf to produce a configure script.
AC_PREREQ([2.57])
AC_INIT([mbxmlutils],[1.1.0],[friedrich.at.gc@googlemail.com])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([xsd2html/addDoxygenToXsd.xsl])
AC_CONFIG_HEADERS([config.h])
LT_INIT([win32-dll])
AC_PROG_CC
AC_PROG_CXX
AC_LANG([C++])

dnl By autoscan
AC_CHECK_FUNCS([dup2])
AC_CHECK_FUNCS([getcwd])
AC_CHECK_FUNCS([memmove])
AC_CHECK_FUNCS([strchr]) 
AC_CHECK_FUNCS([malloc]) 
AC_CHECK_HEADERS([stddef.h])
AC_CHECK_TYPES([ptrdiff_t])
AC_C_INLINE
AC_HEADER_STDBOOL
AC_TYPE_SIZE_T

AC_CANONICAL_HOST

AC_CHECK_HEADERS([unordered_map unordered_set])

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
# check for pthread and include if found
AC_CHECK_LIB(pthread, pthread_mutexattr_init, LIBS="$LIBS -lpthread")
# user supplied mkoctfile program
AC_ARG_WITH([mkoctfile],
  [  --with-mkoctfile=PATHTOMKOCTFILE  The 'mkoctfile' program.],
  [mkoctfile="$withval"],[mkoctfile="no"])
# user supplied libraries for static linking against octave
# add static version of octave DLD (*.oct) files here
AC_ARG_WITH([octave-dld],
  [  --with-octave-dld=STATICLIBARCHIV  Add STATICLIBARCHIV between -loctinterp and -loctave liker flags. (This is only used if static octave DLD (*.oct) files should be included)],
  [STATICOCTAVEDLDLIB="$withval"],[STATICOCTAVEDLDLIB=""])
# get mkoctfile program
if test "$mkoctfile" == "no"; then
  AC_PATH_PROG([mkoctfile],[mkoctfile],[no],[$prefix/bin:$PATH])
fi
test $mkoctfile = "no" && AC_MSG_ERROR([mkoctfile not found])
AC_SUBST([MKOCTFILE],[$mkoctfile])
# get mkoctfile compiler flags
DOS2UNIX="cat"
if test "$cross_compiling" == "yes"; then
  DOS2UNIX="dos2unix"
fi
MKOCTFILE_CFLAGS="$(echo $($MKOCTFILE -p INCFLAGS | $DOS2UNIX) | sed -re 's|\\|/|g')"
MKOCTFILE_LIBS="$(echo \
  $(unset RLD_FLAG; $MKOCTFILE -p RLD_FLAG | $DOS2UNIX) \
  $(unset LFLAGS; $MKOCTFILE -p LFLAGS | $DOS2UNIX) \
  $(unset LIBOCTINTERP; $MKOCTFILE -p LIBOCTINTERP | $DOS2UNIX) \
  $STATICOCTAVEDLDLIB \
  $(unset LIBOCTAVE; $MKOCTFILE -p LIBOCTAVE | $DOS2UNIX) \
  $(unset LIBCRUFT; $MKOCTFILE -p LIBCRUFT | $DOS2UNIX) \
  $(unset FFTW_LIBS; $MKOCTFILE -p FFTW_LIBS | $DOS2UNIX) \
  $(unset BLAS_LIBS; $MKOCTFILE -p BLAS_LIBS | $DOS2UNIX) \
  $(unset FLIBS; $MKOCTFILE -p FLIBS | $DOS2UNIX) \
  $(unset LIBS; $MKOCTFILE -p LIBS | $DOS2UNIX) \
  | sed -re 's|\\|/|g')"

if test "$cross_compiling" == "yes"; then
  # convert MKOCTFILE_CFLAGS to unix path if cross compiling
  RERUN=0
  while test $RERUN -eq 0; do
    WINPATH=$(echo $MKOCTFILE_CFLAGS | sed -re "s|.*-I(@<:@^ @:>@+).*|\1|")
    UNIXPATH=$(readlink -f $(winepath -u $WINPATH))
    MKOCTFILE_CFLAGS=$(echo $MKOCTFILE_CFLAGS | sed -re "s|(.*)-I@<:@^ @:>@+(.*)|\1-X$UNIXPATH\2|")
    echo $MKOCTFILE_CFLAGS | grep "\-I" &> /dev/null
    RERUN=$?
  done
  MKOCTFILE_CFLAGS=$(echo $MKOCTFILE_CFLAGS | sed -re "s|-X|-I|g")

  # convert MKOCTFILE_LIBS to unix path if cross compiling
  RERUN=0
  while test $RERUN -eq 0; do
    WINPATH=$(echo $MKOCTFILE_LIBS | sed -re "s|.*-L(@<:@^ @:>@+).*|\1|")
    UNIXPATH=$(readlink -f $(winepath -u $WINPATH))
    MKOCTFILE_LIBS=$(echo $MKOCTFILE_LIBS | sed -re "s|(.*)-L@<:@^ @:>@+(.*)|\1-X$UNIXPATH\2|")
    echo $MKOCTFILE_LIBS | grep "\-L" &> /dev/null
    RERUN=$?
  done
  MKOCTFILE_LIBS=$(echo $MKOCTFILE_LIBS | sed -re "s|-X|-L|g")
fi

AC_SUBST([MKOCTFILE_CFLAGS])
AC_SUBST([MKOCTFILE_LIBS])
# try to compile a octave stand alone program
SAVE_CPPFLAGS=$CPPFLAGS; CPPFLAGS="$CPPFLAGS $MKOCTFILE_CFLAGS"
SAVE_LIBS=$LIBS; LIBS="$MKOCTFILE_LIBS $LIBS"
AC_CHECKING(compiling a octave program)
AC_LINK_IFELSE([AC_LANG_PROGRAM([[
#include <stdlib.h>
#include <cstdlib>
#include <unistd.h>
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

# from now on only libtool is used for linking
LDFLAGS="$LDFLAGS -no-undefined"

AC_CONFIG_FILES([Makefile xsd2html/Makefile mbxmlutilstinyxml/Makefile mbxmlutils/Makefile mbxmlutilstinyxml.pc mbxmlutils.pc mbxmlutils/env.h mbxmlutils/octave/Makefile mbxmlutils/check/Makefile])
AC_CONFIG_FILES([xsd2html/mbxmlutils-xsd2html],[chmod +x xsd2html/mbxmlutils-xsd2html])

AC_OUTPUT
