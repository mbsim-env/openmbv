dnl Process this file with autoconf to produce a configure script.
AC_PREREQ([2.57])
AC_INIT([openmbvcppinterface],[3.0.0],[mafriedrich@users.berlios.de])
AC_CONFIG_HEADER(config.h)
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([openmbvcppinterface/body.h])
AC_PROG_LIBTOOL
AC_PROG_CXX
AC_PROG_CXXCPP
AC_LANG([C++])

dnl Check for hdf5serie
PKG_CHECK_MODULES(HDF5SERIE, hdf5serie)

dnl Check for doxygen
AC_ARG_WITH([doxygenpath],
  [  --with-doxygenpath=PATH  The path to the 'doxygen' program.],
  [doxygenpath="$withval"],[doxygenpath=""])
AC_PATH_PROG([doxygen],[doxygen],[no],[$doxygenpath:$PATH])
if test $doxygen = "no" ; then
  AC_MSG_RESULT([can't generate API documentation. 'doxygen' is required.])
fi
#DOXYGENDEPTAGFILES=$(pkg-config deppackage --variable=DOXYGENTAGFILES)
#AC_SUBST([DOXYGENDEPTAGFILES])

AC_CONFIG_FILES([Makefile openmbvcppinterface.pc openmbvcppinterface/Makefile
  openmbvcppinterface/check/Makefile doc/Doxyfile])

AC_OUTPUT
