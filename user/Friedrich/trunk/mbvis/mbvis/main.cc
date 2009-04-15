#include "config.h"
#include <QtGui/QApplication>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
  QApplication app(argc, argv);
  MainWindow mainWindow(argc, argv);
  mainWindow.show();
  return app.exec();
}
