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

dnl force strict standard conformance
CPPFLAGS="$CPPFLAGS -Wall -Werror -Wno-long-long"
CFLAGS="$CFLAGS -std=iso9899:1990"

AC_SUBST([DEFINES])

dnl Check for hdf5serie
PKG_CHECK_MODULES(HDF5SERIE, hdf5serie)

PKG_CHECK_MODULES(MBXMLUTILSTINYXML, mbxmlutilstinyxml)

dnl Include path for Boost
AC_ARG_WITH([boost-inc],
  [  --with-boost-inc=PATH  The path Boost include directory.],
  [CPPFLAGS="$CPPFLAGS -I$withval"; DEFINES="$DEFINES -I$withval"])

dnl Check for Boost file locking
AC_CHECK_HEADER([boost/interprocess/sync/file_lock.hpp],
  [AC_DEFINE([HAVE_BOOST_FILE_LOCK], [1], [Boost interprocess file lock found]) DEFINES="$DEFINES -DHAVE_BOOST_FILE_LOCK"])

dnl Check for SWIG
AC_ARG_WITH([swigpath], [  --with-swig-path=PATH  The path to the swig program.],
  [swigpath="$withval"],[swigpath=""])
AC_PATH_PROG([swig],[swig],[no],[$swigpath:$PATH])
AM_CONDITIONAL([COND_SWIG], [test "_$swig" != "_no"])
dnl Check for octave
AC_ARG_WITH([mkoctfilepath], [  --with-mkoctfile-path=PATH  The path to the mkoctfile program.],
  [mkoctfilepath="$withval"],[mkoctfilepath=""])
AC_PATH_PROG([mkoctfile],[mkoctfile],[no],[$mkoctfilepath:$PATH])
AM_CONDITIONAL([COND_MKOCTFILE], [test "_$mkoctfile" != "_no"])
if test "_$mkoctfile" != "_no"; then
  AC_SUBST([OCTAVE_LIBS], ["$($mkoctfile -p LFLAGS) $($mkoctfile -p OCTAVE_LIBS)"])
  AC_SUBST([OCTAVE], [$(dirname $mkoctfile)/octave])
fi
dnl Check for python
AC_ARG_WITH([pythonversion], [  --with-python-version=VER  The version of python to build for (default: 2.7).],
  [pythonversion="$withval"],[pythonversion="2.7"])
PKG_CHECK_MODULES(PYTHON, python-$pythonversion, [pythonfound=yes], [pythonfound=no])
AM_CONDITIONAL([COND_PYTHON], [test "_$pythonfound" == "_yes"])
if test "_$pythonfound" == "_yes"; then
  AC_SUBST([PYTHONBIN], ["$(pkg-config --variable=exec_prefix python-$pythonversion)/bin/python$pythonversion"])
fi

dnl Check for doxygen
AC_ARG_WITH([doxygenpath],
  [  --with-doxygenpath=PATH  The path to the 'doxygen' program.],
  [doxygenpath="$withval"],[doxygenpath=""])
AC_PATH_PROG([doxygen],[doxygen],[no],[$doxygenpath:$PATH])
if test $doxygen = "no" ; then
  AC_MSG_RESULT([can't generate API documentation. 'doxygen' is required.])
fi

AC_SUBST([MBXMLUTILSSCHEMA],[$(pkg-config mbxmlutils --variable SCHEMADIR)])
AM_CONDITIONAL([COND_XML], [test "_$MBXMLUTILSSCHEMA" != "_"])

LDFLAGS="$LDFLAGS -no-undefined -Wl,-no-undefined"

AC_CONFIG_FILES([Makefile doc/Makefile openmbvcppinterface.pc openmbvcppinterface/Makefile
  openmbvcppinterface/check/Makefile doc/Doxyfile schema/Makefile swig/Makefile swig/octave/Makefile swig/python/Makefile])
AC_CONFIG_FILES([swig/octave/octavetest.m], [chmod +x swig/octave/octavetest.m])
AC_CONFIG_FILES([swig/python/pythontest.py], [chmod +x swig/python/pythontest.py])

AC_OUTPUT
