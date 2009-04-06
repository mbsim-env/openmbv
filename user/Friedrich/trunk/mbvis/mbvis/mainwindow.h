#ifndef _MAINWINDOW_H_
#define _MAINWINDOW_H_

#include <QtGui/QMainWindow>
#include <QtGui/QTreeWidget>
#include <string>
#include <Inventor/nodes/SoSeparator.h>

class MainWindow : public QMainWindow {
  Q_OBJECT
  protected:
    QTreeWidget *objectList;
    SoSeparator *sceneRoot;
    void openFile(std::string fileName);
  protected slots:
    void objectListClicked();
    void openFileDialog();
  public:
    MainWindow(int argc, char *argv[]);
};

#endif
