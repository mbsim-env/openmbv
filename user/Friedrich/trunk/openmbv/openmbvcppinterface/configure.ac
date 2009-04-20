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

PKG_CHECK_MODULES(HDF5SERIE, hdf5serie)

AC_CONFIG_FILES([Makefile openmbvcppinterface.pc openmbvcppinterface/Makefile
  openmbvcppinterface/check/Makefile])

AC_OUTPUT
