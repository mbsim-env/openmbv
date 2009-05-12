dnl Process this file with autoconf to produce a configure script.
AC_PREREQ([2.57])
AC_INIT([openmbv],[3.0.0],[mafriedrich@users.berlios.de])
AC_CONFIG_HEADER([config.h])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([openmbv/object.h])
AC_PROG_LIBTOOL
AC_PROG_CXX
AC_PROG_CXXCPP
AC_LANG([C++])

dnl By autoscan
AC_CHECK_FUNCS([memmove])
AC_CHECK_FUNCS([strchr])
AC_CHECK_FUNCS([pow])
AC_CHECK_FUNCS([sqrt])
AC_CHECK_HEADERS([stddef.h])
AC_CHECK_TYPES([ptrdiff_t])
AC_C_INLINE
AC_HEADER_STDBOOL
AC_TYPE_SIZE_T

dnl check host
AC_CANONICAL_HOST
if test "_$host_os" = "_mingw32"; then
  LDFLAGS="-mwindows $LDFLAGS"
  WIN32=yes
fi
AM_CONDITIONAL([COND_WIN32], [test "_$WIN32" = "_yes"])

dnl Qt
PKG_CHECK_MODULES(QT, [QtCore >= 4.4 QtGui >= 4.4 QtSvg >= 4.4 QtWebKit >= 4.4])
dnl get moc, uic and rcc
MOC=$(pkg-config --variable moc_location QtCore)
UIC=$(pkg-config --variable uic_location QtCore)
RCC=${MOC%*/*}/rcc
AC_SUBST([MOC])
AC_SUBST([UIC])
AC_SUBST([RCC])
DIR1=$(pkg-config --variable libdir QtCore)
DIR2=$(pkg-config --variable prefix QtCore)
dnl check for static qsvg imageformats plugin
qsvg=0
AC_CHECK_FILE([$DIR1/plugins/imageformats/libqsvg.a], [qsvg=1; QT_LIBS="-L$DIR1/plugins/imageformats -lqsvg $QT_LIBS"])
AC_CHECK_FILE([$DIR2/plugins/imageformats/libqsvg.a], [qsvg=1; QT_LIBS="-L$DIR2/plugins/imageformats -lqsvg $QT_LIBS"])
if test $qsvg -eq 1; then
  AC_DEFINE_UNQUOTED([STATICQSVGPLUGIN],[],[Use static qsvg plugin])
fi
dnl check for static qsvgicon iconengines plugin
qsvgicon=0
AC_CHECK_FILE([$DIR1/plugins/iconengines/libqsvgicon.a], [qsvgicon=1; QT_LIBS="-L$DIR1/plugins/iconengines -lqsvgicon $QT_LIBS"])
AC_CHECK_FILE([$DIR2/plugins/iconengines/libqsvgicon.a], [qsvgicon=1; QT_LIBS="-L$DIR2/plugins/iconengines -lqsvgicon $QT_LIBS"])
if test $qsvgicon -eq 1; then
  AC_DEFINE_UNQUOTED([STATICQSVGICONPLUGIN],[],[Use static qsvgicon plugin])
fi

dnl Coin
PKG_CHECK_MODULES(COIN, Coin)

dnl SoQt
AC_ARG_WITH(soqt_prefix,
  [  --with-soqt-prefix=PFX  Prefix where SoQt is installed],
  [SOQT_CFLAGS="-I$withval/include"
   SOQT_LIBS="-L$withval/lib -lSoQt"],
  [SOQT_CFLAGS="-I$prefix/include"
   SOQT_LIBS="-L$prefix/lib -lSoQt"])
OLD=$CPPFLAGS; CPPFLAGS="$CPPFLAGS $SOQT_CFLAGS $QT_CFLAGS"
AC_CHECK_HEADERS([Inventor/Qt/SoQt.h], [],
  [AC_MSG_ERROR([Can not find SoQt c++ header Inventor/Qt/SoQt.h])])
CPPFLAGS=$OLD
AC_SUBST([SOQT_CFLAGS])
AC_SUBST([SOQT_LIBS])

dnl HDF5Serie
PKG_CHECK_MODULES(HDF5SERIE, hdf5serie)

dnl qwt >= 5
AC_ARG_WITH(qwt_inc_prefix,
  [  --with-qwt-inc-prefix=PFX  Prefix where qwt headers are installed],
  [QWT_CFLAGS="-I$withval"],
  [QWT_CFLAGS="-I$prefix/include"])
AC_ARG_WITH(qwt_lib_prefix,
  [  --with-qwt-lib-prefix=PFX  Prefix where qwt library is installed],
  [QWT_LIBS="-L$withval -lqwt"],
  [QWT_LIBS="-L$prefix/lib -lqwt"])
OLD=$CPPFLAGS; CPPFLAGS="$CPPFLAGS $QWT_CFLAGS $QT_CFLAGS"
AC_CHECK_HEADERS([qwt_wheel.h], [QWT=yes],
  [AC_MSG_WARN([Can not find qwt >= 5! Some functionality will not work!]); QWT=no])
CPPFLAGS=$OLD
AC_SUBST([QWT_CFLAGS])
AC_SUBST([QWT_LIBS])
AM_CONDITIONAL([COND_QWT], [test "_$QWT" = "_yes"])

dnl XML
PKG_CHECK_MODULES(MBXMLUTILS, [mbxmlutils >= 1.0.0], [HAVE_MBXMLUTILS=yes], [HAVE_MBXMLUTILS=no])
AM_CONDITIONAL([COND_MBXMLUTILS], [test "_$HAVE_MBXMLUTILS" = "_yes"])
if test "_$HAVE_MBXMLUTILS" = "_yes"; then
  AC_SUBST([MBXMLUTILSSCHEMA],[$(pkg-config mbxmlutils --variable SCHEMADIR)])
fi

AC_CONFIG_FILES([Makefile openmbvtinyxml/Makefile openmbv/Makefile schema/Makefile schema/openmbv.xsd doc/Makefile])

AC_OUTPUT
