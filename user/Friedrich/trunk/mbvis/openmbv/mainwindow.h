#ifndef _MAINWINDOW_H_
#define _MAINWINDOW_H_

#include "config.h"
#include <QtGui/QMainWindow>
#include <QtGui/QTreeWidget>
#include <QtGui/QTextEdit>
#include <QtGui/QSpinBox>
#include <QtGui/QActionGroup>
#include <QtGui/QLabel>
#include <QtGui/QStatusBar>
#include <QTimer>
#include <QTime>
#include <string>
#include "body.h"
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoEventCallback.h>
#include <Inventor/SoOffscreenRenderer.h>
#include "SoQtMyViewer.h"
#ifdef HAVE_QWT5_QWT_WHEEL_H
#  include <qwt5/qwt_wheel.h>
#endif

class MainWindow : public QMainWindow {
  Q_OBJECT
  private:
    static MainWindow *instance;
    enum ViewSide { top, bottom, front, back, right, left };
    enum Mode { no, rotate, translate, zoom };
    enum Animation { stop, play, lastFrame };
    Mode mode;
    SoGetBoundingBoxAction *bboxAction;
  protected:
    SoSeparator *sceneRootBBox;
    QTreeWidget *objectList;
    QTextEdit *objectInfo;
    QSpinBox *frameSB;
    void openFile(std::string fileName);
    SoQtMyViewer *glViewer;
    void viewParallel(ViewSide side);
    SoSeparator *sceneRoot;
    QTimer *animTimer;
    QTime *time;
    QDoubleSpinBox *speedSB;
    int animStartFrame;
    QActionGroup *animGroup;
    QSlider *timeSlider;
    double deltaTime;
    SoSFUInt32 *frame;
    QStatusBar *statusBar;
    QLabel *fps;
    QTime *fpsTime;
#ifdef HAVE_QWT5_QWT_WHEEL_H
    QwtWheel *speedWheel;
    double oldSpeed;
#endif
  protected slots:
    void objectListClicked();
    void openFileDialog();
    void aboutOpenMBV();
    void updateFrame(int frame_) { frame->setValue(frame_); }
    void viewAllSlot() { glViewer->viewAll(); }
    void toggleCameraTypeSlot() { glViewer->toggleCameraType(); }
    void viewTopSlot() { viewParallel(top); }
    void viewBottomSlot() { viewParallel(bottom); }
    void viewFrontSlot() { viewParallel(front); }
    void viewBackSlot() { viewParallel(back); }
    void viewRightSlot() { viewParallel(right); }
    void viewLeftSlot() { viewParallel(left); }
    void setObjectInfo(QTreeWidgetItem* current) { if(current) objectInfo->setHtml(((Object*)current)->getInfo()); }
    void frameSBSetRange(int min, int max) { frameSB->setRange(min, max); } // because QAbstractSlider::setRange is not a slot
    void animationButtonSlot(QAction* act);
    void heavyWorkSlot();
    void speedChanged(double value);
    void speedWheelChanged(double value);
    void speedWheelPressed();
    void speedWheelReleased();
    void exportAsPNG(SoOffscreenRenderer &myrendere, std::string fileName, bool transparent, float red, float green, float blue);
    void exportCurrentAsPNG();
    void exportSequenceAsPNG();
  public:
    MainWindow(int argc, char *argv[]);
    static MainWindow*const getInstance() { return instance; }
    bool soQtEventCB(const SoEvent *const event);
    static void frameSensorCB(void *data, SoSensor*);
    static void fpsSensorCBS(void *data, SoSensor*) { MainWindow::getInstance()->fpsSensorCB(); }
    void fpsSensorCB();
    SoSeparator *getSceneRootBBox() { return sceneRootBBox; }
    QSlider *getTimeSlider() { return timeSlider; }
    double &getDeltaTime() { return deltaTime; }
    double getSpeed() { return speedSB->value(); }
    QStatusBar *getStatusBar() { return statusBar; }
    SoSFUInt32 *getFrame() { return frame; }
};

#endif
