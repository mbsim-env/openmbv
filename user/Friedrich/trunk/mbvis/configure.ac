dnl Process this file with autoconf to produce a configure script.
AC_PREREQ([2.57])
AC_INIT([mbvis],[3.0.0],[mafriedrich@users.berlios.de])
AC_CONFIG_HEADER([config.h])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([mbvis/object.h])
AC_PROG_LIBTOOL
AC_PROG_CXX
AC_PROG_CXXCPP
AC_LANG([C++])

dnl By autoscan
AC_CHECK_FUNCS([memmove])
AC_CHECK_FUNCS([strchr])
AC_CHECK_HEADERS([stddef.h])
AC_CHECK_TYPES([ptrdiff_t])
AC_C_INLINE
AC_HEADER_STDBOOL
AC_TYPE_SIZE_T

dnl Qt
AT_WITH_QT

dnl Coin
PKG_CHECK_MODULES(COIN, Coin)

dnl SoQt
AC_ARG_WITH(soqt_prefix,
  [  --with-soqt-prefix=PFX  Prefix where SoQt is installed],
  [SOQT_CFLAGS="-I$withval/include"
   SOQT_LDFLAGS="-L$withval/lib"
   SOQT_LIBS="-lSoQt"],
  [SOQT_CFLAGS="-I$prefix/include"
   SOQT_LDFLAGS="-L$prefix/lib"
   SOQT_LIBS="-lSoQt"])
OLD=$CPPFLAGS; CPPFLAGS="$CPPFLAGS $SOQT_CFLAGS $QT_CPPFLAGS"
AC_CHECK_HEADERS([Inventor/Qt/SoQt.h], [],
  [AC_MSG_ERROR([Can not find SoQt c++ header Inventor/Qt/SoQt.h])])
CPPFLAGS=$OLD
AC_SUBST([SOQT_CFLAGS])
AC_SUBST([SOQT_LDFLAGS])
AC_SUBST([SOQT_LIBS])

AC_CONFIG_FILES([Makefile tinyxml/Makefile mbvis/Makefile])

AC_OUTPUT
