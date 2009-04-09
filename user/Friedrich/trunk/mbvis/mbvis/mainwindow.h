#ifndef _MAINWINDOW_H_
#define _MAINWINDOW_H_

#include <QtGui/QMainWindow>
#include <QtGui/QTreeWidget>
#include <QtGui/QTextEdit>
#include <string>
#include "body.h"
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoEventCallback.h>
#include "SoQtMyViewer.h"

class MainWindow : public QMainWindow {
  Q_OBJECT
  private:
    enum ViewSide { top, bottom, front, back, right, left };
    enum Mode { no, rotate, translate, zoom };
    static Mode mode;
    SoGetBoundingBoxAction *bboxAction;
  protected:
    QTreeWidget *objectList;
    QTextEdit *objectInfo;
    void openFile(std::string fileName);
    SoQtMyViewer *glViewer;
    void viewParallel(ViewSide side);
    SoSeparator *sceneRoot;
  protected slots:
    void objectListClicked();
    void openFileDialog();
    void aboutMBVis();
    void updateFrame(int frame) { Body::frame->setValue(frame); }
    void viewAllSlot() { glViewer->viewAll(); }
    void toggleCameraTypeSlot() { glViewer->toggleCameraType(); }
    void viewTopSlot() { viewParallel(top); }
    void viewBottomSlot() { viewParallel(bottom); }
    void viewFrontSlot() { viewParallel(front); }
    void viewBackSlot() { viewParallel(back); }
    void viewRightSlot() { viewParallel(right); }
    void viewLeftSlot() { viewParallel(left); }
    void setObjectInfo(QTreeWidgetItem* current) { if(current) objectInfo->setHtml(((Object*)current)->getInfo()); }
  public:
    MainWindow(int argc, char *argv[]);
    bool soQtEventCB(const SoEvent *const event);
    static void frameSensorCB(void *data, SoSensor*);
    static SoSeparator *sceneRootBBox;
};

#endif
