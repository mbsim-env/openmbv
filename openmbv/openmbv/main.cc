/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

  This library is free software; you can redistribute it and/or 
  modify it under the terms of the GNU Lesser General Public 
  License as published by the Free Software Foundation; either 
  version 2.1 of the License, or (at your option) any later version. 
   
  This library is distributed in the hope that it will be useful, 
  but WITHOUT ANY WARRANTY; without even the implied warranty of 
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
  Lesser General Public License for more details. 
   
  You should have received a copy of the GNU Lesser General Public 
  License along with this library; if not, write to the Free Software 
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
*/

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif
#include "config.h"
#include <clocale>
#include <cassert>
#include <cfenv>
#include <QApplication>
#include <QFileInfo>
#include <QSettings>
#include "mainwindow.h"
#include "utils.h"
#include <boost/filesystem.hpp>
#include "set_current_path.h"
#include <mbxmlutilshelper/utils.h>
#ifndef _WIN32
#  include "qt-unix-signals/sigwatch.h"
#endif

using namespace std;

namespace {
  #ifndef NDEBUG
    void myQtMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg);
  #endif
}

int main(int argc, char *argv[]) {
#ifdef _WIN32
  SetConsoleCP(CP_UTF8);
  SetConsoleOutputCP(CP_UTF8);
#endif
  MBXMLUtils::handleFPE();
  setlocale(LC_ALL, "C");
  QLocale::setDefault(QLocale::C);

  // check for errors during ObjectFactory
  string errorMsg(OpenMBV::ObjectFactory::getAndClearErrorMsg());
  if(!errorMsg.empty()) {
    cerr<<"The following errors occured during the pre-main code of the OpenMBVC++Interface object factory:"<<endl;
    cerr<<errorMsg;
    cerr<<"Exiting now."<<endl;
    return 1;
  }

  list<string> arg;
  for(int i=1; i<argc; i++)
    arg.emplace_back(argv[i]);

  // current directory and adapt paths
  boost::filesystem::path dirFile;
  if(!arg.empty())
    dirFile=*(--arg.end());
  boost::filesystem::path newCurrentPath;
  if(auto i=std::find(arg.begin(), arg.end(), "--CC"); !dirFile.empty() && i!=arg.end()) {
    if(boost::filesystem::is_directory(dirFile))
      newCurrentPath=dirFile;
    else
      newCurrentPath=dirFile.parent_path();
    arg.erase(i);
  }
  if(auto i=std::find(arg.begin(), arg.end(), "-C"); i!=arg.end()) {
    auto i2=i; i2++;
    if(boost::filesystem::is_directory(*i2))
      newCurrentPath=*i2;
    else
      newCurrentPath=boost::filesystem::path(*i2).parent_path();
    arg.erase(i);
    arg.erase(i2);
  }
  SetCurrentPath currentPath(newCurrentPath);
  for(auto a : {"--wst", "--camera", "--headlight"})
    if(auto i=std::find(arg.begin(), arg.end(), a); i!=arg.end()) {
      auto i2=i; i2++;
      *i2=currentPath.adaptPath(*i2).string();
    }
  for(auto i=arg.rbegin(); i!=arg.rend(); ++i)
    if(currentPath.existsInOrg(*i))
      *i=currentPath.adaptPath(*i).string();

  // check parameters
  list<string>::iterator i, i2;

  i=find(arg.begin(), arg.end(), "-v");
  i2=find(arg.begin(), arg.end(), "--verbose");
  if(i==arg.end() && i2==arg.end())
    fmatvec::Atom::setCurrentMessageStream(fmatvec::Atom::Info, std::make_shared<bool>(false));

  // help
  i=find(arg.begin(), arg.end(), "-h");
  i2=find(arg.begin(), arg.end(), "--help");
  if(i!=arg.end() || i2!=arg.end()) {
        // 12345678901234567890123456789012345678901234567890123456789012345678901234567890
    cout<<"OpenMBV - Open Multi Body Viewer"<<endl
        <<""<<endl
        <<"Copyright (C) 2009 Markus Friedrich <friedrich.at.gc@googlemail.com"<<endl
        <<"This is free software; see the source for copying conditions. There is NO"<<endl
        <<"warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."<<endl
        <<""<<endl
        <<"Licensed under the GNU Lesser General Public License (LGPL)"<<endl
        <<""<<endl
        <<"Usage: openmbv [-h|--help] [-v|--verbose] [--play|--lastframe] [--speed <factor>]"<<endl
        <<"               [--closeall]"<<endl
        <<"               [--wst <file>] [--camera <file>] [--fullscreen]"<<endl
        <<"               [--geometry WIDTHxHEIGHT+X+Y] [--nodecoration]"<<endl
        <<"               [--headlight <file>]"<<endl
        <<"               [-C <dir/file>|--CC]"<<endl
        <<"               [--maximized] [<dir>|<file>] [<dir>|<file>] ..."<<endl
        // 12345678901234567890123456789012345678901234567890123456789012345678901234567890
        <<""<<endl
        <<"-h|--help          Shows this help"<<endl
        <<"-v|--verbose       Print informational messages to stdout"<<endl
        <<"--play             Start animation after loading"<<endl
        <<"--speed            Set the animation speed"<<endl
        <<"--lastframe        View last frame after loading"<<endl
        <<"--closeall         Start with all widgets closed except scene widget"<<endl
        <<"--nodecoration     Disable the window decoration (Titel/Border/...)"<<endl
        <<"--geometry         Set the main(window) geometry"<<endl
        <<"--wst              Load the given (main)window state file"<<endl
        <<"--camera           Load the given camera file (*.iv)"<<endl
        <<"                   (Must be of type OrthographicCamera or PerspectiveCamera)"<<endl
        // 12345678901234567890123456789012345678901234567890123456789012345678901234567890
        <<"--headlight        Load the given head light file (*.iv)"<<endl
        <<"                   (Must be of type DirectionalLight)"<<endl
        <<"--fullscreen       Start in full screen mode"<<endl
        <<"-C <dir/file>      Change current to dir to <dir>/dir of <file> first."<<endl
        <<"                   All arguments are still relative to the original current dir."<<endl
        <<"--CC               Change current dir to dir of last <dir> argument or dir of last <file> argument."<<endl
        <<"                   All arguments are still relative to the original current dir."<<endl
        <<"--maximized        Show window maximized on startup."<<endl
        <<"<dir>              Open/Load all [^.]+\\.ombvx files"<<endl
        <<"                   in <dir>. Only fully preprocessed xml files are allowd."<<endl
        <<"                   <dir> and <file> must be the last arguments."<<endl
        <<"<file>             Open/Load <file>. Only fully preprocessed xml files"<<endl
        <<"                   are allowd."<<endl
        <<"                   <dir> and <file> must be the last arguments."<<endl
        <<""<<endl
        <<"Note:"<<endl
        <<""<<endl
        <<"In contrast to Coin3D VBO (Vertex Buffer Object) is disabled per default in"<<endl
        <<"OpenMBV. You can enable it by setting the environment variable COIN_VBO=1."<<endl
        <<""<<endl
        <<"If you experience crashes at startup regarding the OpenGL context, try"<<endl
        <<"setting the envvar OPENMBV_NO_MULTISAMPLING=2."<<endl;
        // 12345678901234567890123456789012345678901234567890123456789012345678901234567890
    return 0;
  }

  char moduleName[2048];
#ifdef _WIN32
  GetModuleFileName(nullptr, moduleName, sizeof(moduleName));
#else
  size_t s=readlink("/proc/self/exe", moduleName, sizeof(moduleName));
  moduleName[s]=0; // null terminate
#endif
  QCoreApplication::setLibraryPaths(QStringList(QFileInfo(moduleName).absolutePath())); // do not load plugins from buildin defaults

  auto argSaved=arg; // save arguments (QApplication removes all arguments known by Qt)
  QApplication app(argc, argv);
#ifndef NDEBUG
  qInstallMessageHandler(myQtMessageHandler);
#endif
  arg=argSaved; // restore arguments
#ifndef _WIN32
  UnixSignalWatcher sigwatch;
  sigwatch.watchForSignal(SIGHUP);
  sigwatch.watchForSignal(SIGINT);
  sigwatch.watchForSignal(SIGTERM);
  QObject::connect(&sigwatch, &UnixSignalWatcher::unixSignal, &app, &QApplication::quit);
#endif

  app.setOrganizationName(OpenMBVGUI::AppSettings::organization);
  app.setApplicationName(OpenMBVGUI::AppSettings::application);
  app.setOrganizationDomain(OpenMBVGUI::AppSettings::organizationDomain);
  QSettings::setDefaultFormat(OpenMBVGUI::AppSettings::format);
  // Only the standard C locale is supported


  OpenMBVGUI::MainWindow mainWindow(arg);
  mainWindow.show();
  if(mainWindow.getEnableFullScreen()) mainWindow.showFullScreen(); // must be done afer mainWindow.show()
  mainWindow.updateScene(); // must be called after mainWindow.show()
  int ret=app.exec();
  return ret;
}

namespace {
#ifndef NDEBUG
  void myQtMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg) {
    static map<QtMsgType, string> typeStr {
      {QtDebugMsg,    "Debug"},
      {QtWarningMsg,  "Warning"},
      {QtCriticalMsg, "Critical"},
      {QtFatalMsg,    "Fatal"},
      {QtInfoMsg,     "Info"},
    };
    string category(context.category?context.category:"<nocategory>");
    cerr<<(context.file?context.file:"<nofile>")<<":"<<context.line<<": "<<(context.function?context.function:"<nofunc>")<<": "<<category
        <<": "<<typeStr[type]<<": "<<msg.toStdString()<<endl;
    cerr.flush();
    if(category=="qt.accessibility.atspi")
      return;
    switch(type) {
      case QtDebugMsg:
      case QtInfoMsg:
        break;
      case QtWarningMsg:
      case QtCriticalMsg:
      case QtFatalMsg:
        std::abort();
        break;
    }
  }
#endif
}
