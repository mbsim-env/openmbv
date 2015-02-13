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

AC_CHECK_HEADERS([unordered_map unordered_set])
 
#BEGIN: TEMPORARILY DISABLED: seem to break exception handling and produces valgrind errors
#AC_CHECK_HEADERS([libunwind.h])
#if test $target_cpu == "x86_64"; then
#  UNWIND_LIB_CPU=unwind-x86_64
#else
#  UNWIND_LIB_CPU=unwind-x86
#fi
#AC_CHECK_LIB($UNWIND_LIB_CPU, main, [UNWIND_LIB="-l$UNWIND_LIB_CPU -lunwind"], [UNWIND_LIB=""])
#AC_SUBST([UNWIND_LIB], [$UNWIND_LIB])
#AC_DEFINE_UNQUOTED([HAVE_LIBUNWIND], [$(test "_$UNWIND_LIB" != "_" && echo 1 || echo 0)], [libunwind is available])
#AC_CHECK_HEADERS([cxxabi.h])
#END: TEMPORARILY DISABLED: seem to break exception handling and produces valgrind errors

dnl Include path for Boost
AC_ARG_WITH([boost-inc],
  [  --with-boost-inc=PATH  The path Boost include directory.],
  [CPPFLAGS="$CPPFLAGS -I$withval"; DEFINES="$DEFINES -I$withval"])
  
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
AC_ARG_ENABLE([casadi],[  --disable-casadi   disable casadi support],[usecasadi="no"],[usecasadi="yes"])
if test $usecasadi == yes;  then
  AC_ARG_WITH(casadi_inc_prefix,
    [  --with-casadi-inc-prefix=PFX  Prefix where casadi includes are installed],
    [CASADICPPFLAGS="-I$withval"],
    [CASADICPPFLAGS="-I$prefix/include"])
  AC_ARG_WITH(casadi_lib_prefix,
    [  --with-casadi-lib-prefix=PFX  Prefix where casadi lib is installed],
    [ CASADILDFLAGS="-L$withval" ],
    [ CASADILDFLAGS="-L$prefix/lib" ])
  AC_ARG_WITH(casadi_bin_prefix,
    [  --with-casadi-bin-prefix=PFX  Prefix where casadi octave interface is installed (default: $prefix/bin).
       For relocatable installations this MUST NOT BE set (casadi MUST be installed to $prefix)!
       Must be equal to casadi-lib-prefix on Windows.],
    [ CASADIBIN="$withval" ],
    [ CASADIBIN="$prefix/bin" ])
  AC_DEFINE_UNQUOTED(MBXMLUTILS_CASADI_BIN, ["$(test $CASADIBIN == "$prefix/bin" && echo "bin" || echo $CASADIBIN)"], [absolute or relatvie dir to CasADi.])
  CPPFLAGS="$CPPFLAGS $CASADICPPFLAGS -DWITHOUT_PRE_1_9_X" # do not allow deprecated functions from casadi
  casadi="no"
  AC_CHECK_HEADERS([casadi/symbolic/sx/sx_node.hpp], [casadi="yes"],
    [AC_MSG_RESULT([Can not find casadi c++ header casadi/symbolic/sx/sx_node.hpp])])
  if test $casadi = "yes" ; then
    AC_MSG_CHECKING([for casadi c++ library])
    LDFLAGS="$LDFLAGS $CASADILDFLAGS"
    CASADI_LIBS="-lcasadi"
    CASADI_CPPFLAGS="$CASADICPPFLAGS"
    test "_$host_os" != "_mingw32" && CASADI_LIBS="$CASADI_LIBS -ldl"
    SAVED_LIBS="$LIBS"
    LIBS="$LIBS $CASADI_LIBS"
    AC_SUBST([CASADI_LIBS])
    AC_TRY_LINK([#include <casadi/symbolic/sx/sx_node.hpp>],[CasADi::SX x],[AC_MSG_RESULT([yes]);],
      [casadi="no"; AC_MSG_RESULT([no]); AC_MSG_ERROR([CasADi headers found but can not link with CasADi c++ library])])
    LIBS="$SAVED_LIBS"
  fi
  if test $casadi = "yes" ; then
    AC_MSG_CHECKING([for casadi octave interface (by SWIG)])
    if test -f $CASADIBIN/casadi_interface.oct -a -f $CASADIBIN/casadi.m -a -d $CASADIBIN/@swig_ref; then
      AC_MSG_RESULT([yes])
    else
      AC_MSG_RESULT([no])
      AC_MSG_ERROR([
CasADi headers and c++ libraries found. However MBXMLUtils requires also the
SWIG interface of CasADi for octave. This code is build using the command
"make octave" in the CasADi build directory (SWIG is required. Install it
before running "make octave" using your package manager). Since CasADi has
not build code for installing the octave part of CasADi this must be done
manually. After build the octave interface in CasADi copy the following files:
cp <casadi-build-dir>/lib/casadi_interface.oct $CASADIBIN
cp <casadi-build-dir>/lib/casadi.m $CASADIBIN
cp -r <casadi-build-dir>/lib/@swig_ref $CASADIBIN
Alternatively you can disable the CasADi support using --disable-casadi.
Please note that with CasADi 2.X the octave interface has been dropped. Hence
you have to use CasADi 1.X.
It is recommended to use the CasADi fork from
https://github.com/friedrichatgc/casadi.git
      ])
    fi
  fi
fi

AC_SUBST([CASADI_CPPFLAGS])

dnl check if only mbxmlutilshelper should be installed (necessary for openmbvcppinterface and mbsim)
AC_ARG_ENABLE([only_mbxmlutilshelper],
	    AS_HELP_STRING([--enable-only-mbxmlutilshelper], [Enable only 'helper'-part of the mbxml-utils.]))

AS_IF([test "x$enable_only_mbxmlutilshelper" != "xyes"], [

    dnl Check for xslt
    AC_ARG_WITH([xalan],
      [  --with-xalan=XALAN  The Xalan program.],
      [xalan="$withval"],[xalan="$(which Xalan)"])
    test -x "$xalan" || AC_MSG_ERROR([Xalan not found. Install xalan-c])
    AC_SUBST([XALAN],[$xalan])
    
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
    
    DATADIR="$(eval echo $(eval echo $datadir))"
    AC_SUBST([XSLDIR],[$DATADIR/$PACKAGE/xsl])
    AC_SUBST([OCTAVEDIR],[$DATADIR/$PACKAGE/octave])
    AC_SUBST([XMLDIR],[$DATADIR/$PACKAGE/xml])
    
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

# from now on only libtool is used for linking
LDFLAGS="$LDFLAGS -no-undefined -Wl,--no-undefined"

dnl set conditional variable for the subdirectories in the Makefile.am
AM_CONDITIONAL([NOTONLY_MBXMLUTILSHELPER], [test  "x$enable_only_mbxmlutilshelper" != "xyes"])
AM_CONDITIONAL([MBXMLUTILS_MINGW], [test "_$host_os" = "_mingw32"])

AC_CONFIG_FILES([Makefile xmldoc/Makefile xsd2html/Makefile mbxmlutilshelper/Makefile mbxmlutils/Makefile mbxmlutilsbase.pc mbxmlutilshelper.pc mbxmlutils.pc mbxmlutils/octave/Makefile doc/Makefile doc/Doxyfile])
AC_CONFIG_FILES([xsd2html/mbxmlutils-xsd2html],[chmod +x xsd2html/mbxmlutils-xsd2html])

AC_OUTPUT
