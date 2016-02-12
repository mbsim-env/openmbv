dnl Process this file with autoconf to produce a configure script.
AC_PREREQ([2.57])
AC_INIT([openmbv],[0.0.0],[friedrich.at.gc@googlemail.com]) # Version is not used (see "GET VERSION")
AC_CONFIG_HEADER([config.h])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([openmbv/object.h])
LT_INIT([win32-dll])
AC_PROG_CXX
AC_PROG_CXXCPP
AC_LANG([C++])

dnl By autoscan
AC_CHECK_FUNCS([memmove])
AC_CHECK_FUNCS([strchr])
AC_CHECK_FUNCS([pow])
AC_CHECK_FUNCS([sqrt])
AC_CHECK_FUNCS([setlocale])
AC_CHECK_HEADERS([stddef.h])
AC_CHECK_HEADERS([locale.h])
AC_CHECK_TYPES([ptrdiff_t])
AC_C_INLINE
AC_HEADER_STDBOOL
AC_TYPE_SIZE_T
AC_TYPE_INT32_T

AC_CHECK_HEADERS([boost/filesystem.hpp],[],[AC_MSG_ERROR([Can not find boost/filesystem.hpp])])

AC_ARG_WITH([boost-filesystem-lib],
  [  --with-boost-filesystem-lib=LIBNAME  The library name of boost filesystem excluding prefix/suffix (Default: boost_filesystem).],
  [BOOST_FILESYSTEM_LIB="$withval"], [BOOST_FILESYSTEM_LIB="boost_filesystem"])
AC_SUBST([BOOST_FILESYSTEM_LIB])
AC_ARG_WITH([boost-system-lib],
  [  --with-boost-system-lib=LIBNAME  The library name of boost system excluding prefix/suffix (Default: boost_system).],
  [BOOST_SYSTEM_LIB="$withval"], [BOOST_SYSTEM_LIB="boost_system"])
AC_SUBST([BOOST_SYSTEM_LIB])

dnl check host
AC_CANONICAL_HOST
if test "_$host_os" = "_mingw32"; then
  # -fno-keep-inline-dllexport: requires gcc >= 4.6; and is reauired by Qt to avoid the generatoin of
  # code for inline functions which causes undefines symbols on linkage.
  CPPFLAGS="-fno-keep-inline-dllexport $CPPFLAGS"
  LDFLAGS="-mwindows $LDFLAGS"
  WIN32=yes
fi
AM_CONDITIONAL([COND_WIN32], [test "_$WIN32" = "_yes"])

AC_CHECK_HEADERS([unordered_map])

dnl Qt
PKG_CHECK_MODULES(QT, [QtCore >= 4.4 QtGui >= 4.4 QtSvg >= 4.4 QtWebKit >= 4.4])
dnl get moc, uic and rcc
test -z $MOC && { MOC=$(pkg-config --variable moc_location QtCore) || MOC=moc; }
test -z $UIC && { UIC=$(pkg-config --variable uic_location QtCore) || UIC=uic; }
test -z $RCC && { RCC=$(dirname $(which $MOC))/rcc || RCC=rcc; }
AC_SUBST([MOC])
AC_SUBST([UIC])
AC_SUBST([RCC])

AC_ARG_WITH(qmake, [  --with-qmake=PATH qmake program],
  [QMAKE="$withval"], [QMAKE="qmake"])
QT_PLUGINS=$($QMAKE -query QT_INSTALL_PLUGINS)
AC_SUBST([QT_PLUGINS])

dnl Coin
PKG_CHECK_MODULES(COIN, Coin >= 3)
dnl MBXMLUtilsBase
PKG_CHECK_MODULES(MBXMLUTILSBASE, mbxmlutilsbase)

dnl OpenMBVCppInterface
PKG_CHECK_MODULES(OPENMBVCPPINTERFACE, openmbvcppinterface)

dnl SoQt
PKG_CHECK_MODULES(SOQT, SoQt)

dnl HDF5Serie
PKG_CHECK_MODULES(HDF5SERIE, hdf5serie)

dnl qwt >= 6
AC_ARG_WITH(qwt_inc_prefix,
  [  --with-qwt-inc-prefix=PFX  Prefix where qwt6 headers are installed],
  [QWT_CFLAGS="-I$withval"],
  [QWT_CFLAGS="-I$prefix/include"])
AC_ARG_WITH(qwt_lib_name,
  [  --with-qwt-lib-name=name  Qwt6 library name -l<name> (defaults to qwt)],
  [QWT_LIBNAME="$withval"],
  [QWT_LIBNAME="qwt"])
AC_ARG_WITH(qwt_lib_prefix,
  [  --with-qwt-lib-prefix=PFX  Prefix where qwt6 library is installed],
  [QWT_LIBS="-L$withval -l$QWT_LIBNAME"],
  [QWT_LIBS="-L$prefix/lib -l$QWT_LIBNAME"])
OLDCPPFLAGS=$CPPFLAGS; CPPFLAGS="$CPPFLAGS $QWT_CFLAGS $QT_CFLAGS"
OLDLIBS=$LIBS; LIBS="$LIBS $QWT_LIBS $QT_LIBS"
AC_CHECK_HEADERS([qwt_wheel.h], [QWT=yes], [AC_MSG_WARN([Can not find qwt_wheel.h! Some functionality will not work!]); QWT=no])
if test "_$QWT" == "_yes"; then
AC_CHECK_HEADERS([qwt_system_clock.h], [], [AC_MSG_ERROR([Can not find qwt_system_clock.h! Maybe a qwt5 header is used!])])
AC_TRY_LINK(
  [#include <qwt_system_clock.h>],
  [new QwtSystemClock;],
  [],
  [AC_MSG_ERROR([Can not link with qwt >= 6!])])
fi
CPPFLAGS=$OLDCPPFLAGS
LIBS=$OLDLIBS
AC_SUBST([QWT_CFLAGS])
AC_SUBST([QWT_LIBS])
AM_CONDITIONAL([COND_QWT], [test "_$QWT" = "_yes"])

dnl XML
AC_SUBST([MBXMLUTILSSCHEMADIR],[$(pkg-config mbxmlutils --variable SCHEMADIR)])
AC_SUBST([MBXMLUTILSBINDIR],[$(pkg-config mbxmlutils --variable BINDIR)])
AC_SUBST([MBXMLUTILSDOCDIR],[$(pkg-config mbxmlutils --variable DOCDIR)])
AC_SUBST([MBXMLUTILSINCDIR],[$(pkg-config mbxmlutils --variable includedir)])
AC_SUBST([OPENMBVCPPINTERFACEINCDIR],[$(pkg-config openmbvcppinterface --variable includedir)])

# windres
AC_ARG_WITH(windres,
  [  --with-windres=prog  The windres program (defaults to windres)],
  [WINDRES="$withval"],
  [WINDRES="windres"])
AC_SUBST([WINDRES])

CPPFLAGS="$CPPFLAGS -Wall -Werror -Wno-error=attributes"
LDFLAGS="-no-undefined -Wl,--no-undefined $LDFLAGS"
test "_$host_os" != "_mingw32" && LDFLAGS="$LDFLAGS -Wl,--disable-new-dtags,-rpath,\\\$\$ORIGIN/../lib"

AC_CONFIG_FILES([Makefile openmbv.pc openmbv/Makefile xmldoc/Makefile xmldoc/Doxyfile])

AC_OUTPUT
