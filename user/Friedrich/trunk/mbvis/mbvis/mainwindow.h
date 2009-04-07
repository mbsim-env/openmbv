#ifndef _MAINWINDOW_H_
#define _MAINWINDOW_H_

#include <QtGui/QMainWindow>
#include <QtGui/QTreeWidget>
#include <string>
#include "body.h"
#include <Inventor/nodes/SoSeparator.h>
//#include <Inventor/Qt/viewers/SoQtExaminerViewer.h>
#include "SoQtMyViewer.h"

class MainWindow : public QMainWindow {
  Q_OBJECT
  protected:
    QTreeWidget *objectList;
    SoSeparator *sceneRoot;
    void openFile(std::string fileName);
    //SoQtExaminerViewer *glViewer;
    SoQtMyViewer *glViewer;
    void viewParallel(int i);
  protected slots:
    void objectListClicked();
    void openFileDialog();
    void aboutMBVis();
    void updateFrame(int frame) { Body::frame->setValue(frame); }
    void viewAllSlot() { glViewer->viewAll(); }
    void toggleCameraTypeSlot() { glViewer->toggleCameraType(); }
    void viewTopSlot() { viewParallel(0); }
    void viewBottomSlot() { viewParallel(1); }
    void viewFrontSlot() { viewParallel(2); }
    void viewBackSlot() { viewParallel(3); }
    void viewRightSlot() { viewParallel(4); }
    void viewLeftSlot() { viewParallel(5); }
  public:
    MainWindow(int argc, char *argv[]);
};

#endif
