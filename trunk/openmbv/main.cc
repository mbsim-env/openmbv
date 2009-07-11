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
#include <QtGui/QApplication>
#include <QtPlugin>
#include "mainwindow.h"
#include <list>
#include <string>
#include <algorithm>

using namespace std;

#ifdef STATICQSVGPLUGIN
  Q_IMPORT_PLUGIN(qsvg)
#endif
#ifdef STATICQSVGICONPLUGIN
  Q_IMPORT_PLUGIN(qsvgicon)
#endif

int main(int argc, char *argv[])
{
  QApplication app(argc, argv);

  list<string> arg;
  for(int i=1; i<argc; i++)
    arg.push_back(argv[i]);

  // check parameters
  list<string>::iterator i, i2;
  // help
  i=find(arg.begin(), arg.end(), "-h");
  i2=find(arg.begin(), arg.end(), "--help");
  if(i!=arg.end() || i2!=arg.end()) {
    cout<<"OpenMBV - Open Multi Body Viewer"<<endl
        <<""<<endl
        <<"Copyright (C) 2009 Markus Friedrich <mafriedrich@users.berlios.de"<<endl
        <<"This is free software; see the source for copying conditions. There is NO"<<endl
        <<"warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."<<endl
        <<""<<endl
        <<"Licensed under the GNU General Public License (GPL)"<<endl
        <<""<<endl
        <<"Usage: openmbv [-h|--help] [--play|--lastframe] [--speed <factor>]"<<endl
        <<"               [--topbgcolor #XXXXXX] [--bottombgcolor #XXXXXX] [--closeall]"<<endl
        <<"               [--wst <file>] [--camera <file>] [] [--fullscreen]"<<endl
        <<"               [--geometry WIDTHxHEIGHT+X+Y] [--nodecoration]"<<endl
        <<"               [--headlight <file>] [<dir>|<file>] [<dir>|<file>] ..."<<endl
        <<""<<endl
        <<"If no <dir>|<file> argument is given, '.' is appended automatically."<<endl
        <<""<<endl
        <<"-h|--help        Shows this help"<<endl
        <<"--play           Start animation after loading"<<endl
        <<"--speed          Set the animation speed"<<endl
        <<"--lastframe      View last frame after loading"<<endl
        <<"--topbgcolor     The color on the top of the background (red, green, blue value"<<endl
        <<"                 in hex)"<<endl
        <<"--bottombgcolor  The color on the bottom (see also --topbgcolor)"<<endl
        <<"--closeall       Start with all widgets closed except scene widget"<<endl
        <<"--nodecoration   Disable the window decoration (Titel/Border/...)"<<endl
        <<"--geometry       Set the main(window) geometry"<<endl
        <<"--wst            Load the given (main)window state file"<<endl
        <<"--camera         Load the given camera file (*.iv)"<<endl
        <<"                 (Must be of type OrthographicCamera or PerspectiveCamera)"<<endl
        <<"--headlight      Load the given head light file (*.iv)"<<endl
        <<"                 (Must be of type DirectionalLight)"<<endl
        <<"--fullscreen     Start in full screen mode"<<endl
        <<"<dir>            Open/Load all [^.]+\\.ombv.xml and [^.]+\\.ombv.env.xml files"<<endl
        <<"                 in <dir>"<<endl
        <<"<file>           Open/Load <file>"<<endl;
    if(i!=arg.end()) arg.erase(i); if(i2!=arg.end()) arg.erase(i2);
    return 0;
  }

  MainWindow mainWindow(arg);
  mainWindow.show();
  if(mainWindow.getEnableFullScreen()) mainWindow.showFullScreen(); // must be done afer mainWindow.show()
  mainWindow.updateScene(); // must be called after mainWindow.show()

  return app.exec();
}
