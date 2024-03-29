dnl Process this file with autoconf to produce a configure script.
AC_PREREQ([2.57])
AC_INIT([openmbv],[0.0.0],[friedrich.at.gc@googlemail.com]) # Version is not used (see "GET VERSION")
AC_CONFIG_HEADER([config.h])
AM_INIT_AUTOMAKE([subdir-objects])
AC_CONFIG_SRCDIR([openmbv/object.h])
LT_INIT([win32-dll])
AC_PROG_CXX
AC_PROG_CXXCPP
AC_LANG([C++])

# enable C++11
CXXFLAGS="$CXXFLAGS -std=c++17 -DQT_NO_KEYWORDS"

dnl By autoscan
AC_CHECK_FUNCS([memmove])
AC_CHECK_FUNCS([strchr])
AC_CHECK_FUNCS([pow])
AC_CHECK_FUNCS([sqrt])
AC_CHECK_HEADERS([stddef.h])
AC_CHECK_HEADERS([locale.h])
AC_CHECK_TYPES([ptrdiff_t])
AC_CHECK_FUNCS([putenv])
AC_CHECK_FUNCS([strtol])
AC_C_RESTRICT
AC_C_INLINE
AC_HEADER_STDBOOL
AC_TYPE_SIZE_T
AC_TYPE_INT32_T

dnl Include path for Boost
AC_ARG_WITH([boost-inc],
  [  --with-boost-inc=PATH  The path Boost include directory.],
  [CPPFLAGS="$CPPFLAGS -I$withval"; DEFINES="$DEFINES -I$withval"])

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
  CPPFLAGS="-fno-keep-inline-dllexport $CPPFLAGS -DCOIN_DLL -DSOQT_DLL"
  LDFLAGS="-mwindows $LDFLAGS"
  WIN32=yes
else
  CPPFLAGS="$CPPFLAGS -fPIC"
fi
AM_CONDITIONAL([COND_WIN32], [test "_$WIN32" = "_yes"])

dnl Qt
PKG_CHECK_MODULES(QT, [Qt5Core >= 5.6 Qt5Gui >= 5.6 Qt5Widgets >= 5.6 Qt5Svg >= 5.6])
dnl get moc and uic
test -z $MOC && { MOC=$(pkg-config --variable=host_bins Qt5Core)/moc || MOC=moc; }
test -z $UIC && { UIC=$(pkg-config --variable=host_bins Qt5Core)/uic || UIC=uic; }
AC_SUBST([MOC])
AC_SUBST([UIC])

AC_ARG_WITH(qmake, [  --with-qmake=PATH qmake program],
  [QMAKE="$withval"], [QMAKE="qmake"])
QT_PLUGINS=$($QMAKE -query QT_INSTALL_PLUGINS)
AC_SUBST([QT_PLUGINS])
AC_SUBST([QT_PLUGINSSTYLES_DATA], [$(ls -1 $QT_PLUGINS/styles/*.so 2> /dev/null | tr '\n' ' ')])
AC_SUBST([QT_PLUGINSPLATFORMTHEMES_DATA], [$(ls -1 $QT_PLUGINS/platformthemes/*.so 2> /dev/null | tr '\n' ' ')])

dnl Coin
PKG_CHECK_MODULES(COIN, Coin >= 3)

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
OLDCPPFLAGS=$CPPFLAGS; CPPFLAGS="$CPPFLAGS $QWT_CFLAGS $QT_CFLAGS -std=c++17"
OLDLIBS=$LIBS; LIBS="$LIBS $QWT_LIBS $QT_LIBS"
AC_CHECK_HEADERS([qwt_wheel.h], [], [AC_MSG_ERROR([Can not find qwt_wheel.h!])])
AC_CHECK_HEADERS([qwt_system_clock.h], [], [AC_MSG_ERROR([Can not find qwt_system_clock.h! Maybe a qwt5 header is used!])])
AC_TRY_LINK(
  [#include <qwt_system_clock.h>],
  [new QwtSystemClock;],
  [],
  [AC_MSG_ERROR([Can not link with qwt >= 6!])])
CPPFLAGS=$OLDCPPFLAGS
LIBS=$OLDLIBS
AC_SUBST([QWT_CFLAGS])
AC_SUBST([QWT_LIBS])

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
LDFLAGS="$LDFLAGS $LDFLAGS_LIBTOOL -no-undefined -Wl,--no-undefined"
test "_$host_os" != "_mingw32" && LDFLAGS="$LDFLAGS -Wl,--disable-new-dtags,-rpath,\\\$\$ORIGIN/../lib"

AC_CONFIG_FILES([Makefile openmbv.pc openmbv/Makefile xmldoc/Makefile xmldoc/Doxyfile])

hardcode_into_libs=no # do not add hardcoded libdirs to ltlibraries
hardcode_libdir_flag_spec_CXX= # do not add hardcodeed libdirs to ltbinaries
AC_OUTPUT
