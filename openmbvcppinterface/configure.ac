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

AM_CONDITIONAL([COND_MINGW], [test "_$host_os" = "_mingw32"])

dnl Include path for Boost
AC_ARG_WITH([boost-inc],
  [  --with-boost-inc=PATH  The path Boost include directory.],
  [CPPFLAGS="$CPPFLAGS -I$withval"; DEFINES="$DEFINES -I$withval"])

dnl Check for Boost file locking
AC_CHECK_HEADER([boost/interprocess/sync/file_lock.hpp],
  [AC_DEFINE([HAVE_BOOST_FILE_LOCK], [1], [Boost interprocess file lock found]) DEFINES="$DEFINES -DHAVE_BOOST_FILE_LOCK"])

dnl Check for optional boost type_trains and static_assert
AC_CHECK_HEADERS([boost/type_traits.hpp])

dnl Check for SWIG
AC_ARG_WITH([swigpath], [  --with-swigpath=PATH  The path to the swig program.],
  [swigpath="$withval"],[swigpath=""])
AC_PATH_PROG([swig],[swig],[no],[${swigpath:-$PATH}])
AM_CONDITIONAL([COND_SWIG], [test "_$swig" != "_no"])
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

dnl Check for octave
AC_ARG_WITH([mkoctfile], [  --with-mkoctfile=PROG  The 'mkoctfile' program (Use 'no' to disable the octave interface).],
  [mkoctfile="$withval"],[mkoctfile="no"])
if test "$mkoctfile" == "no"; then
  AC_PATH_PROG([mkoctfile],[mkoctfile],[no],[$prefix/bin:$PATH])
fi
AM_CONDITIONAL([COND_MKOCTFILE], [test "_$mkoctfile" != "_no"])
if test "_$mkoctfile" != "_no"; then
  AC_SUBST([OCTAVE], [$(dirname $mkoctfile)/octave$(test "$cross_compiling" == "yes" && echo .exe)])
  OCTAVE_LIBS="$(echo $($mkoctfile -p LFLAGS | $DOS2UNIX) $($mkoctfile -p OCTAVE_LIBS | $DOS2UNIX) | sed -re 's|\\|/|g')"
  OCTAVE_FLAGS="$(echo $($mkoctfile -p INCFLAGS | $DOS2UNIX) | sed -re 's|\\|/|g')"
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
fi

dnl Check for python
AC_ARG_WITH([pythonversion], [  --with-pythonversion=VER  The version of python to build for (default: 2.7).],
  [pythonversion="$withval"],[pythonversion="2.7"])
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
AC_ARG_WITH([javajnicflags],[  --with-javajnicflags=FLAGS  c-Compile flags for java JNI (e.g -I...)], [javajnicflags="$withval"],[javajnicflags=""])
AC_PATH_PROG([java], [java], [no],[${javapath:-$PATH}])
if test "_$javac" == "_no"; then
  AC_PATH_PROG([javac],[gcj],[no],[$PATH])
  if test "_$javac" != "_no"; then
    javac="$javac -C" # append -C = create bytecode
  else
    AC_PATH_PROG([javac],[javac],[no],[$PATH]) # try javac
  fi
fi
AC_SUBST([JAVA], [$java])
AC_SUBST([JAVAC], [$javac])
AC_SUBST([JAVA_JNICFLAGS], [$javajnicflags])
AM_CONDITIONAL([COND_JAVA], [test "_$java" != "_no" -a "_$javac" != "_no"])

dnl Check for doxygen
AC_ARG_WITH([doxygenpath],
  [  --with-doxygenpath=PATH  The path to the 'doxygen' program.],
  [doxygenpath="$withval"],[doxygenpath=""])
AC_PATH_PROG([doxygen],[doxygen],[no],[${doxygenpath:-$PATH}])
if test $doxygen = "no" ; then
  AC_MSG_RESULT([can't generate API documentation. 'doxygen' is required.])
fi

AC_SUBST([MBXMLUTILSSCHEMA],[$(pkg-config mbxmlutils --variable SCHEMADIR)])
AM_CONDITIONAL([COND_XML], [test "_$MBXMLUTILSSCHEMA" != "_"])

LDFLAGS="$LDFLAGS -no-undefined -Wl,-no-undefined"

AC_CONFIG_FILES([Makefile doc/Makefile openmbvcppinterface.pc openmbvcppinterface/Makefile
  openmbvcppinterface/check/Makefile doc/Doxyfile schema/Makefile swig/Makefile swig/octave/Makefile swig/python/Makefile swig/java/Makefile swig/octave/octavetest.m swig/python/pythontest.py swig/java/matlabtest.m])
AC_CONFIG_FILES([swig/octave/octavetest.sh], [chmod +x swig/octave/octavetest.sh])
AC_CONFIG_FILES([swig/python/pythontest.sh], [chmod +x swig/python/pythontest.sh])
AC_CONFIG_FILES([swig/java/javatest.sh], [chmod +x swig/java/javatest.sh])

AC_OUTPUT
