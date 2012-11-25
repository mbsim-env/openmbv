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

LDFLAGS="-Wl,--no-undefined $LDFLAGS"
AC_CHECK_HEADERS([unordered_map])

dnl GET VERSION from svn
SVNVERSION1=$(svnversion -n . https://openmbv.googlecode.com/svn/trunk/openmbv)
SVNVERSION2=$(svnversion -n . http://openmbv.googlecode.com/svn/trunk/openmbv)
if test ${#SVNVERSION1} -lt ${#SVNVERSION2}; then
  SVNVERSION=$SVNVERSION1
else
  SVNVERSION=$SVNVERSION2
fi
AC_DEFINE_UNQUOTED([SVNVERSION],["$SVNVERSION"],[The svn version/revision])

dnl Qt
PKG_CHECK_MODULES(QT, [QtCore >= 4.4 QtGui >= 4.4 QtSvg >= 4.4 QtWebKit >= 4.4])
dnl get moc, uic and rcc
test -z $MOC && { MOC=$(pkg-config --variable moc_location QtCore) || MOC=moc; }
test -z $UIC && { UIC=$(pkg-config --variable uic_location QtCore) || UIC=uic; }
test -z $RCC && { RCC=$(dirname $(which $MOC))/rcc || RCC=rcc; }
AC_SUBST([MOC])
AC_SUBST([UIC])
AC_SUBST([RCC])
test -d "$QTLIBDIR" || QTLIBDIR=$(pkg-config --variable libdir QtCore)
test -d "$QTPREFIX" || QTPREFIX=$(pkg-config --variable prefix QtCore)

dnl Coin
PKG_CHECK_MODULES(COIN, Coin)

dnl OpenMBVCppInterface
PKG_CHECK_MODULES(OPENMBVCPPINTERFACE, openmbvcppinterface)

dnl SoQt
PKG_CHECK_MODULES(SOQT, SoQt)

dnl HDF5Serie
PKG_CHECK_MODULES(HDF5SERIE, hdf5serie)

dnl qwt >= 5
AC_ARG_WITH(qwt_inc_prefix,
  [  --with-qwt-inc-prefix=PFX  Prefix where qwt headers are installed],
  [QWT_CFLAGS="-I$withval"],
  [QWT_CFLAGS="-I$prefix/include"])
AC_ARG_WITH(qwt_lib_name,
  [  --with-qwt-lib-lib=name  QWt library name -l<name> (defaults to qwt)],
  [QWT_LIBNAME="$withval"],
  [QWT_LIBNAME="qwt"])
AC_ARG_WITH(qwt_lib_prefix,
  [  --with-qwt-lib-prefix=PFX  Prefix where qwt library is installed],
  [QWT_LIBS="-L$withval -l$QWT_LIBNAME"],
  [QWT_LIBS="-L$prefix/lib -l$QWT_LIBNAME"])
OLDCPPFLAGS=$CPPFLAGS; CPPFLAGS="$CPPFLAGS $QWT_CFLAGS $QT_CFLAGS"
OLDLIBS=$LIBS; LIBS="$LIBS $QWT_LIBS $QT_LIBS"
AC_CHECK_HEADERS([qwt_wheel.h], [QWT=yes],
  [AC_MSG_WARN([Can not find qwt >= 5! Some functionality will not work!]); QWT=no])
if test "_$QWT" == "_yes"; then
AC_TRY_LINK(
  [#include <qwt_plot_grid.h>],
  [new QwtPlotGrid;],
  [],
  [AC_MSG_ERROR([Can not link with qwt >= 5!])])
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
AC_SUBST([OPENMBVCPPINTERFACEINCDIR])

# windres
AC_ARG_WITH(windres,
  [  --with-windres=prog  The windres program (defaults to windres)],
  [WINDRES="$withval"],
  [WINDRES="windres"])
AC_SUBST([WINDRES])

CPPFLAGS="$CPPFLAGS -Wall -Werror"
CXXFLAGS="$CXXFLAGS -Wno-non-virtual-dtor"
LDFLAGS="$LDFLAGS -no-undefined"

AC_CONFIG_FILES([Makefile openmbv.pc openmbv/Makefile doc/Makefile doc/Doxyfile])

AC_OUTPUT
