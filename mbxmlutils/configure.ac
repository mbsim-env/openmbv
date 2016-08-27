dnl Process this file with autoconf to produce a configure script.
AC_PREREQ([2.57])
AC_INIT([mbxmlutils],[1.3.0],[friedrich.at.gc@googlemail.com])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([xsd2html/addDoxygenToXsd.xsl])
AC_CONFIG_HEADERS([config.h])
LT_INIT([win32-dll])
AC_PROG_CC
AC_PROG_CXX
AC_LANG([C++])

# enable C++11
CXXFLAGS="$CXXFLAGS -std=c++11"

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

AC_CANONICAL_SYSTEM

dnl Include path for Boost
AC_ARG_WITH([boost-inc],
  [  --with-boost-inc=PATH  The path Boost include directory.],
  [CPPFLAGS="$CPPFLAGS -I$withval"; DEFINES="$DEFINES -I$withval"])
AC_SUBST([DEFINES])
  
AC_CHECK_HEADERS([boost/filesystem.hpp],[],[AC_MSG_ERROR([boost/filesystem.hpp not found.])])

AC_ARG_WITH([boost-filesystem-lib],
  [  --with-boost-filesystem-lib=LIBNAME  The library name of boost filesystem excluding prefix/suffix (Default: boost_filesystem).],
  [BOOST_FILESYSTEM_LIB="$withval"], [BOOST_FILESYSTEM_LIB="boost_filesystem"])
AC_SUBST([BOOST_FILESYSTEM_LIB])
AC_ARG_WITH([boost-system-lib],
  [  --with-boost-system-lib=LIBNAME  The library name of boost system excluding prefix/suffix (Default: boost_system).],
  [BOOST_SYSTEM_LIB="$withval"], [BOOST_SYSTEM_LIB="boost_system"])
AC_SUBST([BOOST_SYSTEM_LIB])

AC_SUBST([DATADIR], ["$(eval echo $(eval echo $datadir))"])

dnl shared libraries
if test "_$host_os" != "_mingw32"; then
  LIBDL="-ldl"
fi
AC_SUBST([LIBDL])

dnl Check for casadi 
PKG_CHECK_MODULES(CASADI, [casadi >= 3.0.0])
CASADI_PREFIX_DIR="$($PKG_CONFIG --variable=prefix casadi)"
AC_SUBST([CASADI_PREFIX_DIR])

dnl Check for SWIG
AC_ARG_WITH([swigpath], [  --with-swigpath=PATH  The path to the swig program.],
  [swigpath="$withval"],[swigpath=""])
AC_PATH_PROG([swig],[swig],[no],[${swigpath:-$PATH}])
AM_CONDITIONAL([COND_SWIG], [test "_$swig" != "_no"])
if test "_$swig" != "_no"; then
  SWIGVERSION=$($swig -version | sed -rne "s/^SWIG Version (.*)$/\1/p")
  AC_MSG_NOTICE([Using SWIG version $SWIGVERSION])
  if test $(echo $SWIGVERSION | sed -re "s/^([[0-9]]+)\..*/\1/") -lt 3; then
    AC_MSG_ERROR([SWIG >= 3 needed])
  fi
else
  AC_MSG_ERROR([SWIG not found])
fi

dnl check if only mbxmlutilshelper should be installed (necessary for openmbvcppinterface and mbsim)
AC_ARG_ENABLE([only_mbxmlutilshelper],
	    AS_HELP_STRING([--enable-only-mbxmlutilshelper], [Enable only 'helper'-part of the mbxml-utils.]))

AS_IF([test "x$enable_only_mbxmlutilshelper" != "xyes"], [
    if test "$cross_compiling" == "yes"; then
      AC_SUBST([XC_EXEC_PREFIX],[wine])
    fi

    dnl Check for xslt
    AC_ARG_WITH([xalan],
      [  --with-xalan=XALAN  The Xalan program.],
      [xalan="$withval"],[xalan="$(which Xalan)"])
    test -x "$xalan" || AC_MSG_ERROR([Xalan not found. Install xalan-c])
    AC_SUBST([XALAN],[$xalan])
    
    dnl Check for octave
    # user supplied mkoctfile program
    AC_ARG_WITH([mkoctfile],
      [  --with-mkoctfile=PATHTOMKOCTFILE  The 'mkoctfile' program.],
      [mkoctfile="$withval"],[mkoctfile="no"])
    # get mkoctfile program
    if test "_$mkoctfile" == "_no"; then
      AC_PATH_PROG([mkoctfile],[mkoctfile],[no],[$prefix/bin:$PATH])
    fi
    test "_$mkoctfile" = "_no" && AC_MSG_ERROR([mkoctfile not found])
    # get mkoctfile compiler flags
    MKOCTFILE_CFLAGS="$(echo $($XC_EXEC_PREFIX $mkoctfile -p INCFLAGS | dos2unix) | sed -re 's|\\|/|g')"
    MKOCTFILE_LIBS="$(echo \
      $(unset RLD_FLAG; $XC_EXEC_PREFIX $mkoctfile -p RLD_FLAG | dos2unix) \
      $(unset LFLAGS; $XC_EXEC_PREFIX $mkoctfile -p LFLAGS | dos2unix) \
      $(unset LIBOCTINTERP; $XC_EXEC_PREFIX $mkoctfile -p LIBOCTINTERP | dos2unix) \
      $(unset LIBOCTAVE; $XC_EXEC_PREFIX $mkoctfile -p LIBOCTAVE | dos2unix) \
      $(unset LIBCRUFT; $XC_EXEC_PREFIX $mkoctfile -p LIBCRUFT | dos2unix) \
      $(unset FFTW_LIBS; $XC_EXEC_PREFIX $mkoctfile -p FFTW_LIBS | dos2unix) \
      $(unset BLAS_LIBS; $XC_EXEC_PREFIX $mkoctfile -p BLAS_LIBS | dos2unix) \
      $(unset FLIBS; $XC_EXEC_PREFIX $mkoctfile -p FLIBS | dos2unix) \
      $(unset LIBS; $XC_EXEC_PREFIX $mkoctfile -p LIBS | dos2unix) \
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
    
    DATADIR="$(eval echo $(eval echo $datadir))"
    AC_SUBST([XSLDIR],[$DATADIR/$PACKAGE/xsl])
    AC_SUBST([OCTAVEDIR],[$DATADIR/$PACKAGE/octave])
    AC_SUBST([XMLDIR],[$DATADIR/$PACKAGE/xml])
    
    # check for python
    AC_ARG_ENABLE([python], [--enable-python  Build the python evaluator. This evalutor is ALPHA hence it must be enabled explicitly.])
    AM_CONDITIONAL([COND_PYTHON], [test "x$enable_python" == "xyes"])
    if test "x$enable_python" == "xyes"; then
      PYTHON_MODULE=python-2.7
      PKG_CHECK_MODULES([PYTHON], [$PYTHON_MODULE])
      NUMPY_INCDIR="$($PKG_CONFIG --variable=libdir $PYTHON_MODULE)/python$($PKG_CONFIG --modversion $PYTHON_MODULE)/site-packages/numpy/core/include"
      if test ! -d $NUMPY_INCDIR; then
        AC_MSG_ERROR([Numpy header not found in directory $NUMPY_INCDIR])
        AC_SUBST([NUMPY_INCDIR])
      fi
    fi
])

AC_SUBST([SCHEMADIR],[$DATADIR/$PACKAGE/schema])

dnl Check for doxygen
AC_ARG_WITH([doxygenpath],
  [  --with-doxygenpath=PATH  The path to the 'doxygen' program.],
  [doxygenpath="$withval"],[doxygenpath=""])
AC_PATH_PROG([doxygen],[doxygen],[no],[${doxygenpath:-$PATH}])
if test $doxygen = "no" ; then
  AC_MSG_RESULT([can not generate API documentation. 'doxygen' is required.])
fi

dnl Check for xerces-c
PKG_CHECK_MODULES(XERCESC, xerces-c,[],[AC_MSG_ERROR([ERROR xerces-c not found])])

PKG_CHECK_MODULES(FMATVEC, fmatvec)

CPPFLAGS="$CPPFLAGS -Wall -Werror -Wno-error=sign-compare -Wno-sign-compare -Wno-delete-non-virtual-dtor"
if test "_$host_os" != "_mingw32"; then
  CPPFLAGS="$CPPFLAGS -fPIC"
fi

# from now on only libtool is used for linking
LDFLAGS="$LDFLAGS -no-undefined -Wl,--no-undefined"
test "_$host_os" != "_mingw32" && LDFLAGS="$LDFLAGS -Wl,--disable-new-dtags,-rpath,\\\$\$ORIGIN/../lib"

dnl set conditional variable for the subdirectories in the Makefile.am
AM_CONDITIONAL([NOTONLY_MBXMLUTILSHELPER], [test  "x$enable_only_mbxmlutilshelper" != "xyes"])
AM_CONDITIONAL([MBXMLUTILS_MINGW], [test "_$host_os" = "_mingw32"])

AC_CONFIG_FILES([Makefile xmldoc/Makefile xsd2html/Makefile mbxmlutilshelper/Makefile mbxmlutils/Makefile mbxmlutilshelper.pc mbxmlutils.pc mbxmlutils/octave/Makefile mbxmlutils/python/Makefile doc/Makefile doc/doxyfile mbxmlutils/pyeval-config.h])
AC_CONFIG_FILES([xsd2html/mbxmlutils-xsd2html],[chmod +x xsd2html/mbxmlutils-xsd2html])

AC_OUTPUT
