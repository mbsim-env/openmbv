dnl Process this file with autoconf to produce a configure script.
AC_PREREQ([2.57])
AC_INIT([mbxmlutils],[1.3.0],[friedrich.at.gc@googlemail.com])
AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([xsd2html/addDoxygenToXsd.xsl])
AC_CONFIG_HEADERS([config.h])
LT_INIT([win32-dll])
AC_PROG_CC
AC_PROG_CXX
AC_LANG([C++])

# enable C++11
CXXFLAGS="$CXXFLAGS -std=c++17"

dnl By autoscan
AC_CHECK_FUNCS([dup2])
AC_CHECK_FUNCS([getcwd])
AC_CHECK_FUNCS([memmove])
AC_CHECK_FUNCS([strchr]) 
AC_CHECK_HEADERS([stddef.h])
AC_CHECK_TYPES([ptrdiff_t])
AC_C_INLINE
AC_HEADER_STDBOOL
AC_TYPE_SIZE_T
AC_CHECK_FUNCS([floor])
AC_CHECK_FUNCS([pow])
# AC_CHECK_FUNCS([putenv]) cannot be added to to Windows build
AC_CHECK_FUNCS([sqrt])
AC_FUNC_ERROR_AT_LINE
AC_TYPE_UINT64_T

# windres
AC_ARG_WITH(windres,
  [  --with-windres=prog  The windres program (defaults to windres)],
  [WINDRES="$withval"],
  [WINDRES="windres"])
AC_SUBST([WINDRES])

# for compile version
OLDCXXFLAGS="$CXXFLAGS"
CXXFLAGS="$CXXFLAGS -Wno-error=class-memaccess"
AC_COMPILE_IFELSE([AC_LANG_SOURCE([int main() {}])], [NOWARNING="-Wno-class-memaccess"], [NOWARNING=""])
CXXFLAGS="$OLDCXXFLAGS"
AC_SUBST([NOWARNING])

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
AC_ARG_WITH([boost-regex-lib],
  [  --with-boost-regex-lib=LIBNAME  The library name of boost regex excluding prefix/suffix (Default: boost_regex).],
  [BOOST_REGEX_LIB="$withval"], [BOOST_REGEX_LIB="boost_regex"])
AC_SUBST([BOOST_REGEX_LIB])

# check for debug build
DEBUGBUILD=1
echo $CFLAGS $CXXFLAGS $CPPFLAGS | grep "\-DNDEBUG" > /dev/null && DEBUGBUILD=0
AC_SUBST([DEBUGBUILD])

dnl shared libraries
if test "_$host_os" != "_mingw32"; then
  LIBDL="-ldl"
fi
AC_SUBST([LIBDL])

dnl Check for xerces-c
PKG_CHECK_MODULES(XERCESC, xerces-c,[],[AC_MSG_ERROR([ERROR xerces-c not found])])

PKG_CHECK_MODULES(FMATVEC, fmatvec)

dnl Check for SWIG
AC_ARG_WITH([swigpath], [  --with-swigpath=PATH  The path to the swig program.],
  [swigpath="$withval"],[swigpath=""])
AC_PATH_PROG([swig],[swig],[no],[${swigpath:-$PATH}])
if test "_$swig" == "_no"; then
  AC_MSG_ERROR([SWIG not found])
fi
SWIGVERSION=$($swig -version | sed -rne "s/^SWIG Version (.*)$/\1/p")
AC_MSG_NOTICE([Using SWIG version $SWIGVERSION])
SWIGVERSION_=$[[$(echo $SWIGVERSION | sed -re "s/^([0-9]+)\.([0-9]+)\.([0-9]+)$/\1*1000000+\2*1000+\3/")]]
if test $SWIGVERSION_ -lt $[[4*1000000+0*1000+1]]; then
  if test $[[$(cat /etc/debian_version 2> /dev/null | sed -re "s/^([0-9]+)\.([0-9]+)$/\1*1000+\2/")]] -lt 10003 \
       -o $SWIGVERSION_ -lt $[[3*1000000+0*1000+12]]; then # debian >= 10.3 patches swig 3.0.12
    AC_MSG_ERROR([SWIG >= 4.0.1 is needed.])
  fi
fi

XC_PATH_CONVERT=echo
if test "$cross_compiling" == "yes"; then
  AC_SUBST([XC_EXEC_PREFIX],[wine])
  XC_PATH_CONVERT="winepath -w"
fi
AC_SUBST([XC_PATH_CONVERT])

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
# get mkoctfile version
MBXMLUTILS_OCTAVE_MAJOR_VERSION=$($XC_EXEC_PREFIX $mkoctfile --version | dos2unix | sed -re 's/.*version ([[0-9]]+)\.([[0-9]]+)\.([[0-9]]+).*/\1/')
MBXMLUTILS_OCTAVE_MINOR_VERSION=$($XC_EXEC_PREFIX $mkoctfile --version | dos2unix | sed -re 's/.*version ([[0-9]]+)\.([[0-9]]+)\.([[0-9]]+).*/\2/')
# check octave version
if test $[[$MBXMLUTILS_OCTAVE_MAJOR_VERSION * 1000 + $MBXMLUTILS_OCTAVE_MINOR_VERSION]] -eq 4004; then
  AC_MSG_NOTICE([Using tested octave version 4.4.])
elif test $[[$MBXMLUTILS_OCTAVE_MAJOR_VERSION * 1000 + $MBXMLUTILS_OCTAVE_MINOR_VERSION]] -eq 6002; then
  AC_MSG_NOTICE([Using tested octave version 6.2.])
elif test $[[$MBXMLUTILS_OCTAVE_MAJOR_VERSION * 1000 + $MBXMLUTILS_OCTAVE_MINOR_VERSION]] -eq 7003; then
  AC_MSG_NOTICE([Using tested octave version 7.3.])
elif test $[[$MBXMLUTILS_OCTAVE_MAJOR_VERSION * 1000 + $MBXMLUTILS_OCTAVE_MINOR_VERSION]] -lt 4004; then
  AC_MSG_ERROR([octave versions < 4.4 are not working. Please use a newer version. Tested are version 4.4, 6.2 and 7.3.])
else
  AC_MSG_WARN([octave version $MBXMLUTILS_OCTAVE_MAJOR_VERSION.$MBXMLUTILS_OCTAVE_MINOR_VERSION is not supported/tested.
Supported and tested are versions 4.4, 6.2 and 7.3. Use at your own risk.])
fi
# get mkoctfile compiler flags
MKOCTFILE_CFLAGS="$(echo $($XC_EXEC_PREFIX $mkoctfile -p INCFLAGS | dos2unix) | sed -re 's|\\|/|g')"
MKOCTFILE_LIBDIR="$(echo $($XC_EXEC_PREFIX $mkoctfile -p OCTLIBDIR | dos2unix) | sed -re 's|\\|/|g')"
MKOCTFILE_LIBS="$(echo -L$MKOCTFILE_LIBDIR \
  $(unset RLD_FLAG; $XC_EXEC_PREFIX $mkoctfile -p RLD_FLAG | dos2unix) \
  $(unset LFLAGS; $XC_EXEC_PREFIX $mkoctfile -p LFLAGS | dos2unix) \
  $(unset LDFLAGS; $XC_EXEC_PREFIX $mkoctfile -p LDFLAGS | dos2unix) \
  $(unset LIBOCTINTERP; $XC_EXEC_PREFIX $mkoctfile -p LIBOCTINTERP | dos2unix) \
  $(unset LIBOCTAVE; $XC_EXEC_PREFIX $mkoctfile -p LIBOCTAVE | dos2unix) \
  $(unset LIBCRUFT; $XC_EXEC_PREFIX $mkoctfile -p LIBCRUFT | dos2unix) \
  $(unset FFTW_LIBS; $XC_EXEC_PREFIX $mkoctfile -p FFTW_LIBS | dos2unix) \
  $(unset BLAS_LIBS; $XC_EXEC_PREFIX $mkoctfile -p BLAS_LIBS | dos2unix) \
  $(unset FLIBS; $XC_EXEC_PREFIX $mkoctfile -p FLIBS | dos2unix) \
  $(unset LIBS; $XC_EXEC_PREFIX $mkoctfile -p LIBS | dos2unix) \
  | sed -re 's|\\|/|g')"

MKOCTFILE_OCTAVE_HOME_WIN="$(echo $($XC_EXEC_PREFIX $mkoctfile -p OCTAVE_HOME | dos2unix) | sed -re 's|\\|/|g')"
MKOCTFILE_OCTAVE_HOME_UNIX="$(cygpath -u $MKOCTFILE_OCTAVE_HOME_WIN 2> /dev/null)"
AC_DEFINE_UNQUOTED([MKOCTFILE_OCTAVE_HOME_WIN],  ["$MKOCTFILE_OCTAVE_HOME_WIN"], ["octave home windows"])
AC_DEFINE_UNQUOTED([MKOCTFILE_OCTAVE_HOME_UNIX], ["$MKOCTFILE_OCTAVE_HOME_UNIX"], ["octave home unix"])

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

  # convert MKOCTFILE_LIBDIR to unix path if cross compiling
  MKOCTFILE_LIBDIR=$(readlink -f $(winepath -u $MKOCTFILE_LIBDIR))
fi

AC_SUBST([MKOCTFILE_CFLAGS])
AC_SUBST([MKOCTFILE_LIBS])
AC_SUBST([MKOCTFILE_LIBDIR])
# try to compile a octave stand alone program
SAVE_CPPFLAGS=$CPPFLAGS; CPPFLAGS="$CPPFLAGS $MKOCTFILE_CFLAGS"
SAVE_LIBS=$LIBS; LIBS="$MKOCTFILE_LIBS $LIBS"
AC_CHECKING(compiling a octave program)
AC_LINK_IFELSE([AC_LANG_PROGRAM([[
  #define _USE_MATH_DEFINES
  #include <octave/interpreter.h>
]],[[
  octave::interpreter oct;
]])], AC_MSG_RESULT([done]), AC_MSG_ERROR([can not compile a octave program]))
CPPFLAGS=$SAVE_CPPFLAGS
LIBS=$SAVE_LIBS

# The prefix for octave m and oct file search path can be changed
# using the octave envvar OCTAVE_HOME.

AC_SUBST([OCTAVE], [$(dirname $mkoctfile)/octave-cli$(test "$cross_compiling" == "yes" && echo .exe)])

# check for python
AC_ARG_WITH([pythonversion], [  --with-pythonversion=VER  The version of python to build for (default: $(python3 --version | sed -re "s/.*\<(3\.[[0-9]]+)\..*/\1/")).],
  [pythonversion="$withval"],[pythonversion=$(python3 --version | sed -re "s/.*\<(3\.[[0-9]]+)\..*/\1/")])
PKG_CHECK_MODULES([PYTHON], [python-$pythonversion-embed], [], [
  PKG_CHECK_MODULES([PYTHON], [python-$pythonversion])
])
if test "_$PYTHON_BIN" == "_"; then
  PYTHON_BIN="$(pkg-config --variable=exec_prefix python-$pythonversion)/bin/python$pythonversion"
fi
PYTHON_PREFIX="$(pkg-config --variable=exec_prefix python-$pythonversion)"
PYTHON_LIBDIR=$($XC_EXEC_PREFIX $PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_config_var("LIBDEST"))' | dos2unix)
if test "$cross_compiling" == "yes"; then
  PYTHON_LIBDIR=$(readlink -f $(winepath -u $PYTHON_LIBDIR))
fi
if test "_$host_os" != "_mingw32"; then
  PYTHON_LIBDIR2=$PYTHON_LIBDIR/lib-dynload
else
  PYTHON_LIBDIR2=$PYTHON_LIBDIR/../DLLs
fi
if test -d $PYTHON_LIBDIR/python$pythonversion/site-packages/numpy/core/include; then
  PYTHON_LIBDIR=$PYTHON_LIBDIR/python$pythonversion
  PYTHON_PACKAGES=site-packages
elif test -d $PYTHON_LIBDIR/site-packages/numpy/core/include; then
  PYTHON_LIBDIR=$PYTHON_LIBDIR
  PYTHON_PACKAGES=site-packages
elif test -d /usr/lib/python3/dist-packages/numpy/core/include; then # debian special handling
  PYTHON_LIBDIR=/usr/lib/python3
  PYTHON_PACKAGES=dist-packages
else
  AC_MSG_ERROR([Numpy header not found in directory $PYTHON_LIBDIR/python$pythonversion/site-packages/numpy/core/include or $PYTHON_LIBDIR/site-packages/numpy/core/include])
fi
AC_DEFINE_UNQUOTED([PYTHON_LIBDIR], ["$PYTHON_LIBDIR"], [Python library dir])
AC_DEFINE_UNQUOTED([PYTHON_PREFIX], ["$PYTHON_PREFIX"], [Python prefix])
AC_SUBST([PYTHON_BIN], [$PYTHON_BIN])
AC_SUBST([PYTHON_LIBDIR])
AC_SUBST([PYTHON_LIBDIR2])
AC_SUBST([PYTHON_PACKAGES])
PYTHON_SUBDIR=$(echo $($XC_EXEC_PREFIX $PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_config_var("LIBDEST")[[len(sysconfig.get_config_var("prefix"))+1:]])' | dos2unix) | sed -re 's|\\|/|g')
AC_DEFINE_UNQUOTED([PYTHON_SUBDIR], ["$PYTHON_SUBDIR"], [Python library subdir])
AC_DEFINE_UNQUOTED([PYTHON_VERSION], ["$pythonversion"], [Python version])
AC_SUBST([PYTHON_VERSION],[$pythonversion])

dnl Check for doxygen
AC_ARG_WITH([doxygenpath],
  [  --with-doxygenpath=PATH  The path to the 'doxygen' program.],
  [doxygenpath="$withval"],[doxygenpath=""])
AC_PATH_PROG([doxygen],[doxygen],[no],[${doxygenpath:-$PATH}])
if test $doxygen = "no" ; then
  AC_MSG_RESULT([can not generate API documentation. 'doxygen' is required.])
fi

CPPFLAGS="$CPPFLAGS -Wall -Werror -Wno-error=sign-compare -Wno-sign-compare -Wno-delete-non-virtual-dtor"
if test "_$host_os" != "_mingw32"; then
  CPPFLAGS="$CPPFLAGS -fPIC"
fi

# from now on only libtool is used for linking
LDFLAGS="$LDFLAGS $LDFLAGS_LIBTOOL -no-undefined -Wl,--no-undefined"
test "_$host_os" != "_mingw32" && LDFLAGS="$LDFLAGS -Wl,--disable-new-dtags,-rpath,\\\$\$ORIGIN/../lib"

dnl set conditional variable for the subdirectories in the Makefile.am
AM_CONDITIONAL([COND_WIN32], [test "_$host_os" = "_mingw32"])

AC_CONFIG_FILES([Makefile xmldoc/Makefile xsd2html/Makefile mbxmlutilshelper/Makefile mbxmlutils/Makefile mbxmlutilshelper.pc mbxmlutils.pc mbxmlutils/octave/Makefile mbxmlutils/python/Makefile doc/Makefile doc/doxyfile])
AC_CONFIG_FILES([mbxmlutils/octave/testsymbolic.sh],[chmod +x mbxmlutils/octave/testsymbolic.sh])
AC_CONFIG_FILES([mbxmlutils/python/testsymbolic.sh],[chmod +x mbxmlutils/python/testsymbolic.sh])
AC_CONFIG_FILES([mbxmlutils/measurement.sh],[chmod +x mbxmlutils/measurement.sh])
AC_CONFIG_FILES([xsd2html/mbxmlutils-xsd2html],[chmod +x xsd2html/mbxmlutils-xsd2html])

hardcode_into_libs=no # do not add hardcoded libdirs to ltlibraries
hardcode_libdir_flag_spec_CXX= # do not add hardcodeed libdirs to ltbinaries
AC_OUTPUT
