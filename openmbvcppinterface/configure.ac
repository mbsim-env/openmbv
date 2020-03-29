dnl Process this file with autoconf to produce a configure script.
AC_PREREQ([2.57])
AC_INIT([openmbvcppinterface],[3.1.0],[friedrich.at.gc@googlemail.com])
AC_CONFIG_HEADER(config.h)
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([openmbvcppinterface/body.h])
LT_INIT([win32-dll])
AC_PROG_CXX
AC_PROG_CXXCPP
AC_LANG([C++])

AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T

# enable C++11
CXXFLAGS="$CXXFLAGS -std=c++17"

# for compile version
OLDCXXFLAGS="$CXXFLAGS"
CXXFLAGS="$CXXFLAGS -Wno-error=class-memaccess"
AC_COMPILE_IFELSE([AC_LANG_SOURCE([int main() {}])], [NOWARNING="-Wno-error=class-memaccess"], [NOWARNING=""])
CXXFLAGS="$OLDCXXFLAGS"
AC_SUBST([NOWARNING])

dnl force strict standard conformance
CPPFLAGS="$CPPFLAGS -Wall -Werror -Wno-long-long"
if test "_$host_os" != "_mingw32"; then
  CPPFLAGS="$CPPFLAGS -fPIC"
fi
CFLAGS="$CFLAGS -std=iso9899:1990"

AC_SUBST([DEFINES])

dnl Check for hdf5serie
PKG_CHECK_MODULES(HDF5SERIE, hdf5serie)

PKG_CHECK_MODULES(FMATVEC, fmatvec)

PKG_CHECK_MODULES(MBXMLUTILSHELPER, mbxmlutilshelper)

AM_CONDITIONAL([COND_WIN32], [test "_$host_os" = "_mingw32"])

dnl Include path for Boost
AC_ARG_WITH([boost-inc],
  [  --with-boost-inc=PATH  The path Boost include directory.],
  [CPPFLAGS="$CPPFLAGS -I$withval"; DEFINES="$DEFINES -I$withval"])

dnl Check for Boost file locking
AC_CHECK_HEADER([boost/interprocess/sync/file_lock.hpp],
  [AC_DEFINE([HAVE_BOOST_FILE_LOCK], [1], [Boost interprocess file lock found]) DEFINES="$DEFINES -DHAVE_BOOST_FILE_LOCK"])

AC_ARG_WITH([boost-system-lib],
  [  --with-boost-system-lib=LIBNAME  The library name of boost system excluding prefix/suffix (Default: boost_system).],
  [BOOST_SYSTEM_LIB="$withval"], [BOOST_SYSTEM_LIB="boost_system"])
AC_SUBST([BOOST_SYSTEM_LIB])

AM_CONDITIONAL([COND_CROSSCOMPILE], [test "$cross_compiling" == "yes"])

dnl Check for SWIG
AC_ARG_WITH([swigpath], [  --with-swigpath=PATH  The path to the swig program.],
  [swigpath="$withval"],[swigpath=""])
AC_PATH_PROG([swig],[swig],[no],[${swigpath:-$PATH}])
if test "_$swig" == "_no"; then
  AC_MSG_ERROR([No swig found!])
fi
SWIGVERSION=$($swig -version | sed -rne "s/^SWIG Version (.*)$/\1/p")
AC_MSG_NOTICE([Using SWIG version $SWIGVERSION])
SWIGVERSION_=$[[$(echo $SWIGVERSION | sed -re "s/^([0-9]+)\.([0-9]+)\.([0-9]+)$/\1*1000000+\2*1000+\3/")]]
if test $SWIGVERSION_ -lt $[[3*1000000+0*1000+11]]; then
  AC_MSG_ERROR([SWIG >= 3.0.11])
fi
dnl Check for octave
DOS2UNIX="cat"
if test "$cross_compiling" == "yes"; then
  DOS2UNIX="dos2unix"
fi

dnl Convert prefix to native path
if test "$cross_compiling" == "yes"; then
  native_prefix=$(winepath -w $prefix)
else
  native_prefix=$prefix
fi
AC_SUBST([native_prefix])

XC_PATH_CONVERT=echo
if test "$cross_compiling" == "yes"; then
  AC_SUBST([XC_EXEC_PREFIX],[wine])
  XC_PATH_CONVERT="winepath -w"
fi
AC_SUBST([XC_PATH_CONVERT])

dnl Check for octave
AC_ARG_WITH([mkoctfile], [  --with-mkoctfile=PROG  The 'mkoctfile' program (Use 'no' to disable the octave interface).],
  [mkoctfile="$withval"],[mkoctfile=""])
if test "_$mkoctfile" == "_"; then
  AC_PATH_PROG([mkoctfile],[mkoctfile],[no],[$prefix/bin:$PATH])
fi
if test "_$mkoctfile" == "_no"; then
  AC_MSG_ERROR([mkoctfile not found!])
fi
OCTAVEVERSION=$($XC_EXEC_PREFIX mkoctfile --version 2>&1 | sed -rne "s/^mkoctfile, version (.*)$/\1/p")
AC_MSG_NOTICE([Using octave version $OCTAVEVERSION])
AC_SUBST([OCTAVE], [$(dirname $mkoctfile)/octave$(test "$cross_compiling" == "yes" && echo .exe)])
OCTAVE_LIBS="$(echo $($XC_EXEC_PREFIX $mkoctfile -p LFLAGS | $DOS2UNIX) $($XC_EXEC_PREFIX $mkoctfile -p OCTAVE_LIBS | $DOS2UNIX) | sed -re 's|\\|/|g')"
OCTAVE_FLAGS="$(echo $($XC_EXEC_PREFIX $mkoctfile -p INCFLAGS | $DOS2UNIX) | sed -re 's|\\|/|g')"
if test "$cross_compiling" == "yes"; then
  # convert OCTAVE_FLAGS to unix path if cross compiling
  RERUN=0
  while test $RERUN -eq 0; do
    WINPATH=$(echo $OCTAVE_FLAGS | sed -re "s|.*-I(@<:@^ @:>@+).*|\1|")
    UNIXPATH=$(readlink -f $(winepath -u $WINPATH))
    OCTAVE_FLAGS=$(echo $OCTAVE_FLAGS | sed -re "s|(.*)-I@<:@^ @:>@+(.*)|\1-X$UNIXPATH\2|")
    echo $OCTAVE_FLAGS | grep "\-I" &> /dev/null
    RERUN=$?
  done
  OCTAVE_FLAGS=$(echo $OCTAVE_FLAGS | sed -re "s|-X|-I|g")

  # convert OCTAVE_LIBS to unix path if cross compiling
  RERUN=0
  while test $RERUN -eq 0; do
    WINPATH=$(echo $OCTAVE_LIBS | sed -re "s|.*-L(@<:@^ @:>@+).*|\1|")
    UNIXPATH=$(readlink -f $(winepath -u $WINPATH))
    OCTAVE_LIBS=$(echo $OCTAVE_LIBS | sed -re "s|(.*)-L@<:@^ @:>@+(.*)|\1-X$UNIXPATH\2|")
    echo $OCTAVE_LIBS | grep "\-L" &> /dev/null
    RERUN=$?
  done
  OCTAVE_LIBS=$(echo $OCTAVE_LIBS | sed -re "s|-X|-L|g")
fi
AC_SUBST([OCTAVE_FLAGS])
AC_SUBST([OCTAVE_LIBS])

dnl Check for python
AC_ARG_WITH([pythonversion], [  --with-pythonversion=VER  The version of python to build for (default: $(python3 --version | sed -re "s/.*\<(3\.[[0-9]]+)\..*/\1/")).],
  [pythonversion="$withval"],[pythonversion=$(python3 --version | sed -re "s/.*\<(3\.[[0-9]]+)\..*/\1/")])
PKG_CHECK_MODULES(PYTHON, python-$pythonversion, [pythonfound=yes], [pythonfound=no])
AM_CONDITIONAL([COND_PYTHON], [test "_$pythonfound" == "_yes"])
if test "_$pythonfound" == "_yes"; then
  if test "_$PYTHON_BIN" == "_"; then
    PYTHON_BIN="$(pkg-config --variable=exec_prefix python-$pythonversion)/bin/python$pythonversion"
  fi
  AC_SUBST([PYTHON_BIN])
fi
if test "$cross_compiling" == "yes"; then
  pythonshext=".pyd"
else
  pythonshext=".so"
fi
AC_SUBST([PYTHON_SHEXT], [$pythonshext])

dnl Check for java
AC_ARG_WITH([javapath],  [  --with-javapath=PATH   The path to the java program.],  [javapath="$withval"],[javapath=""])
AC_ARG_WITH([javac], [  --with-javac=PATH  The java bytecode compiler.], [javac="$withval"],[javac="no"])
AC_ARG_WITH([javajniosdir],[  --with-javajniosdir=DIR  Directory where the OS specific jni header (jni_md.h) is located. 
If not given the linux subdir of jni.h is used.], [javajniosdir="$withval"],[javajniosdir=""])
AC_PATH_PROG([java], [java], [no],[${javapath:-$PATH}])
if test "_$javac" == "_no"; then
  AC_PATH_PROG([javac],[javac],[no],[$PATH])
fi
AC_SUBST([JAVA], [$java])
AC_SUBST([JAVAC], [$javac])
javajnidir=$(dirname $(readlink -e $javac))/../include
javajniosdir="${javajniosdir:-$javajnidir/linux}"
AC_SUBST([JAVA_JNICFLAGS], ["-I$javajnidir -I$javajniosdir"])
AM_CONDITIONAL([COND_JAVA], [test "_$java" != "_no" -a "_$javac" != "_no"])
if test "_$java" != "_no" -a "_$javac" != "_no" -a ! -e $javajnidir/jni.h; then
  AC_MSG_ERROR([$javajnidir/jni.h not found!])
fi
# if $javajnidir/jni.h is found than rerun configure if it is changed or removed.
# rerun if changed is not required but should neven happen since its a quite static system file.
# rerun if removed is however usefull to reconfigure automatically on system updates chaning the location of this file
if test -e $javajnidir/jni.h; then
  AC_SUBST([CONFIG_STATUS_DEPENDENCIES], [$javajnidir/jni.h])
fi

dnl Check for doxygen
AC_ARG_WITH([doxygenpath],
  [  --with-doxygenpath=PATH  The path to the 'doxygen' program.],
  [doxygenpath="$withval"],[doxygenpath=""])
AC_PATH_PROG([doxygen],[doxygen],[no],[${doxygenpath:-$PATH}])
if test $doxygen = "no" ; then
  AC_MSG_RESULT([cannot generate API documentation. 'doxygen' is required.])
fi

AC_SUBST([MBXMLUTILSSCHEMA],[$(pkg-config mbxmlutils --variable SCHEMADIR)])

LDFLAGS="$LDFLAGS $LDFLAGS_LIBTOOL -no-undefined -Wl,--no-undefined"
test "_$host_os" != "_mingw32" && LDFLAGS="$LDFLAGS -Wl,--disable-new-dtags,-rpath,\\\$\$ORIGIN/../lib"

AC_CONFIG_FILES([Makefile doc/Makefile openmbvcppinterface.pc openmbvcppinterface/Makefile
  openmbvcppinterface/check/Makefile doc/doxyfile schema/Makefile swig/Makefile swig/octave/Makefile swig/python/Makefile swig/java/Makefile])
AC_CONFIG_FILES([openmbvcppinterface/check/testprog.sh], [chmod +x openmbvcppinterface/check/testprog.sh])
AC_CONFIG_FILES([swig/octave/octavetest.sh], [chmod +x swig/octave/octavetest.sh])
AC_CONFIG_FILES([swig/python/pythontest.sh], [chmod +x swig/python/pythontest.sh])
AC_CONFIG_FILES([swig/java/javatest.sh], [chmod +x swig/java/javatest.sh])

hardcode_into_libs=no # do not add hardcoded libdirs to ltlibraries
hardcode_libdir_flag_spec_CXX= # do not add hardcodeed libdirs to ltbinaries
AC_OUTPUT
