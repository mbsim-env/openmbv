/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include "config.h"
#include <cassert>
#include <cfenv>
#include <QApplication>
#include <QFileInfo>
#include "mainwindow.h"
#ifdef _WIN32
#  include <windows.h>
#endif

using namespace std;

int main(int argc, char *argv[])
{
#ifndef _WIN32
//MISSING Qt seems to generate some FPE, hence disabled  assert(feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW)!=-1);
#endif

  // environment variables
  // Disalbe COIN VBO per default (see --help)
  static char COIN_VBO[11];
  if(getenv("COIN_VBO")==nullptr) putenv(strcpy(COIN_VBO, "COIN_VBO=0"));

  list<string> arg;
  for(int i=1; i<argc; i++)
    arg.emplace_back(argv[i]);

  // check parameters
  list<string>::iterator i, i2;
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
        <<"Licensed under the GNU General Public License (GPL)"<<endl
        <<""<<endl
        <<"Usage: openmbv [-h|--help] [--play|--lastframe] [--speed <factor>]"<<endl
        <<"               [--topbgcolor #XXXXXX] [--bottombgcolor #XXXXXX] [--closeall]"<<endl
        <<"               [--wst <file>] [--camera <file>] [--fullscreen]"<<endl
        <<"               [--geometry WIDTHxHEIGHT+X+Y] [--nodecoration]"<<endl
        <<"               [--headlight <file>] [--olselinewidth <linewidth>]"<<endl
        <<"               [--complexitytype [objectspace|screenspace|boundingbox]]"<<endl
        <<"               [--complexityvalue <value>] [--olsecolor #XXXXXX]"<<endl
        <<"               [--autoreload [<timeout>]] [--transparency 1|2]"<<endl
        <<"               [--maximized] [<dir>|<file>] [<dir>|<file>] ..."<<endl
        // 12345678901234567890123456789012345678901234567890123456789012345678901234567890
        <<""<<endl
        <<"If no <dir>|<file> argument is given, '.' is appended automatically."<<endl
        <<""<<endl
        <<"-h|--help          Shows this help"<<endl
        <<"--play             Start animation after loading"<<endl
        <<"--speed            Set the animation speed"<<endl
        <<"--lastframe        View last frame after loading"<<endl
        <<"--topbgcolor       The color on the top of the background (red, green, blue"<<endl
        <<"                   value in hex)"<<endl
        <<"--bottombgcolor    The color on the bottom (see also --topbgcolor)"<<endl
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
        <<"--olselinewidth    Line width of outlines and shilouette edges"<<endl
        <<"--olsecolor        Color of outlines and shilouette edges"<<endl
        <<"--complexitytype   The complexity type (see inventor SoComplexity.type)"<<endl
        <<"--complexityvalue  The complexity value [0..100] (see inventor"<<endl
        <<"                   SoComplexity.value)"<<endl
        <<"--autoreload       Reload, every <timeout> milli seconds, files that have a"<<endl
        <<"                   newer modification time than before. If <timeout> is omitted"<<endl
        <<"                   a default value is used."<<endl
        <<"--transparency     1 = DELAYED_BLEND (default): fast; independent of graphic"<<endl
        <<"                       card; good results with only opaque objects and objects"<<endl
        <<"                       with similar transparency value."<<endl
        <<"                   2 = SORTED_LAYERS_BLEND (Coin extension): best results;"<<endl
        <<"                       but requires OpenGL extensions by the graphic card."<<endl
        <<"--maximized        Show window maximized on startup."<<endl
        <<"<dir>              Open/Load all [^.]+\\.ombv.xml and [^.]+\\.ombv.env.xml files"<<endl
        <<"                   in <dir>. Only fully preprocessed xml files are allowd."<<endl
        <<"<file>             Open/Load <file>. Only fully preprocessed xml files"<<endl
        <<"                   are allowd."<<endl
        <<""<<endl
        <<"Note:"<<endl
        <<"In contrast to Coin3D VBO (Vertex Buffer Object) is disabled per default in"<<endl
        <<"OpenMBV. You can enable it by setting the environment variable COIN_VBO=1."<<endl;
        // 12345678901234567890123456789012345678901234567890123456789012345678901234567890
    if(i!=arg.end()) arg.erase(i);
    if(i2!=arg.end()) arg.erase(i2);
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
  QApplication app(argc, argv);
  app.setOrganizationName("MBSim-Env");
  // Only the standard C locale is supported
  QLocale::setDefault(QLocale::C);
  setlocale(LC_ALL, "C");

  OpenMBVGUI::MainWindow mainWindow(arg);
  mainWindow.show();
  if(mainWindow.getEnableFullScreen()) mainWindow.showFullScreen(); // must be done afer mainWindow.show()
  mainWindow.updateScene(); // must be called after mainWindow.show()

  return app.exec();
}
