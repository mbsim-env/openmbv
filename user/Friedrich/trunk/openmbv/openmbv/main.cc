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
        <<"Usage: openmbv [-h|--help] [<dir>|<file>] [<dir>|<file>] ..."<<endl
        <<""<<endl
        <<"Calling 'openmbv' without arguments is the same as calling 'openmbv .'"<<endl
        <<""<<endl
        <<"-h|--help  Shows this help"<<endl
        <<"<dir>      Open/Load all [^.]+\\.ombv.xml and [^.]+\\.ombv.env.xml files in <dir>"<<endl
        <<"<file>     Open/Load <file>"<<endl;
    if(i!=arg.end()) arg.erase(i); if(i2!=arg.end()) arg.erase(i2);
    return 0;
  }

  MainWindow mainWindow(arg);
  mainWindow.show();

  return app.exec();
}
