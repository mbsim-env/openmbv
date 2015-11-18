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

#ifndef _OPENMBVGUI_MAINWINDOW_H_
#define _OPENMBVGUI_MAINWINDOW_H_

#include "abstractviewfilter.h"
#include <QtGui/QMainWindow>
#include <QtGui/QTreeWidget>
#include <QtGui/QTextEdit>
#include <QtGui/QSpinBox>
#include <QtGui/QActionGroup>
#include <QtGui/QLabel>
#include <QtGui/QStatusBar>
#include <QtWebKit/QWebView>
#include <QtCore/QTimer>
#include <QtCore/QTime>
#include <string>
#include "body.h"
#include "group.h"
#include "SoSpecial.h"
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoEventCallback.h>
#include <Inventor/engines/SoTransformVec3f.h>
#include "SoTransposeEngine.h"
#include <Inventor/fields/SoSFRotation.h>
#include <Inventor/fields/SoSFUInt32.h>
#include <FXViz/nodes/SoShadowGroup.h>
#include <Inventor/SoOffscreenRenderer.h>
#include "SoQtMyViewer.h"
#include "QTripleSlider.h"
#include <QDropEvent>
#ifdef HAVE_QWT_WHEEL_H
#  include <qwt_wheel.h>
#else
#  include <QtGui/QSlider>
#endif

// If Coin and SoQt is linked as a dll no symbols of this file are exported (for an unknown reason).
// Hence we explicitly export the required symbols. This should be done for all code for a clean Windows build!
#ifdef _WIN32
#  define DLL_PUBLIC __declspec(dllexport)
#else
#  define DLL_PUBLIC
#endif

class QListWidgetItem;

namespace OpenMBVGUI {

class MainWindow : public QMainWindow, virtual public fmatvec::Atom {
  Q_OBJECT
  friend class Body;
  friend class Editor;
  friend class Group;
  friend class CompoundRigidBody;
  private:
    static MainWindow *instance;
    enum ViewSide { top, bottom, front, back, right, left, isometric, dimetric,
                    rotateXpWorld, rotateXmWorld, rotateYpWorld, rotateYmWorld, rotateZpWorld, rotateZmWorld,
                    rotateXpScreen, rotateXmScreen, rotateYpScreen, rotateYmScreen, rotateZpScreen, rotateZmScreen };
    enum Animation { stop, play, lastFrame };
    struct WindowState { bool hasMenuBar, hasStatusBar, hasFrameSlider; };
    SoText2 *timeString;
    double fpsMax;
    QWebView *helpViewerGUI, *helpViewerXML;
    bool enableFullScreen;
    SoTransformVec3f *cameraPosition;
    SoTransposeEngine *cameraOrientation;
    SoSwitch *worldFrameSwitch;
    SoSwitch *engDrawing;
    SoMFColor *engDrawingBGColorSaved, *engDrawingFGColorBottomSaved, *engDrawingFGColorTopSaved;
    SoDrawStyle *olseDrawStyle;
    SoBaseColorHeavyOverride *olseColor;
    int reloadTimeout;
    SoFieldSensor *frameSensor;
  protected:
    SoSepNoPickNoBBox *sceneRootBBox;
    QTreeWidget *objectList;
    AbstractViewFilter *objectListFilter;
    QTextEdit *objectInfo;
    QSpinBox *frameSB, *frameMinSB, *frameMaxSB;
    SoQtMyViewer *glViewer;
    void viewChange(ViewSide side);
    SoShadowGroup *sceneRoot;
    SoComplexity *complexity;
    QTimer *animTimer;
    QTime *time;
    QDoubleSpinBox *speedSB;
    int animStartFrame;
    QActionGroup *animGroup;
    QTripleSlider *timeSlider;
    double deltaTime;
    SoSFUInt32 *frame;
    QLabel *fps;
    QTime *fpsTime;
#ifdef HAVE_QWT_WHEEL_H
    QwtWheel *speedWheel;
#else
    QSlider *speedWheel;
#endif
    double oldSpeed;
    QAction *stopAct, *lastFrameAct, *playAct, *toggleMenuBar, *toggleStatusBar, *toggleFrameSlider, *toggleFullScreen, *toggleDecoration;
    boost::shared_ptr<OpenMBV::Body> openMBVBodyForLastFrame;
    QAction *engDrawingView, *topBGColorAct, *bottomBGColorAct;
    SoMFColor *bgColor, *fgColorTop, *fgColorBottom;
    void help(std::string type, QDialog *helpDialog);
    static void toggleAction(Object *current, QAction *currentAct);
    void execPropertyMenu();
    static void disableBBox(Object *obj);
    static void enableBBoxOfID(Object *obj, const std::string &ID);
  protected slots:
    void objectListClicked();
    void openFileDialog();
    void newFileDialog();
    void guiHelp();
    void xmlHelp();
    void aboutOpenMBV();
    void updateFrame(int frame_) { frame->setValue(frame_); }
    void viewAllSlot() { glViewer->viewAll(); }
    void toggleCameraTypeSlot() { glViewer->toggleCameraType(); }
    void releaseCameraFromBodySlot();
    void showWorldFrameSlot();
    void shadowRenderingSlot();

    void viewTopSlot() { viewChange(top); }
    void viewBottomSlot() { viewChange(bottom); }
    void viewFrontSlot() { viewChange(front); }
    void viewBackSlot() { viewChange(back); }
    void viewRightSlot() { viewChange(right); }
    void viewLeftSlot() { viewChange(left); }
    void viewIsometricSlot() { viewChange(isometric); }
    void viewDimetricSlot() { viewChange(dimetric); }
    void viewRotateXpWorld() { viewChange(rotateXpWorld); }
    void viewRotateXmWorld() { viewChange(rotateXmWorld); }
    void viewRotateYpWorld() { viewChange(rotateYpWorld); }
    void viewRotateYmWorld() { viewChange(rotateYmWorld); }
    void viewRotateZpWorld() { viewChange(rotateZpWorld); }
    void viewRotateZmWorld() { viewChange(rotateZmWorld); }
    void viewRotateXpScreen() { viewChange(rotateXpScreen); }
    void viewRotateXmScreen() { viewChange(rotateXmScreen); }
    void viewRotateYpScreen() { viewChange(rotateYpScreen); }
    void viewRotateYmScreen() { viewChange(rotateYmScreen); }
    void viewRotateZpScreen() { viewChange(rotateZpScreen); }
    void viewRotateZmScreen() { viewChange(rotateZmScreen); }

    void setObjectInfo(QTreeWidgetItem* current) { if(current) objectInfo->setHtml(((Object*)current)->getInfo()); }
    void frameSBSetRange(int min, int max) { frameSB->setRange(min, max); } // because QAbstractSlider::setRange is not a slot
    void heavyWorkSlot();
    void restartPlay();
    void speedWheelChangedD(double value) { speedWheelChanged((int)value); }
    void speedWheelChanged(int value);
    void speedWheelPressed();
    void speedWheelReleased();
    void exportAsPNG(short width, short height, std::string fileName, bool transparent);
    void exportCurrentAsPNG();
    void exportSequenceAsPNG();
    void exportCurrentAsIV();
    void exportCurrentAsPS();
    void helpHomeXML();
    void helpHomeGUI();
    void stopSCSlot();
    void lastFrameSCSlot();
    void playSCSlot();
    void speedUpSlot();
    void speedDownSlot();
    void topBGColor();
    void bottomBGColor();
    void olseColorSlot();
    void olseLineWidthSlot();
    void loadWindowState(std::string filename="");
    void saveWindowState();
    void loadCamera(std::string filename="");
    void saveCamera();
    void toggleMenuBarSlot();
    void toggleStatusBarSlot();
    void toggleFrameSliderSlot();
    void toggleFullScreenSlot();
    void toggleDecorationSlot();
    void collapseItem(QTreeWidgetItem* item);
    void expandItem(QTreeWidgetItem* item);

    // we use our own expandToDepth function since the Qt one does not emit the expand/collapse signal
    void expandToDepth(QTreeWidgetItem *item, int depth) {
      for(int i=0; i<item->childCount(); i++) {
        if(depth>0) {
          if(item->child(i)->childCount()>0) item->child(i)->setExpanded(true); 
        }
        else {
          if(item->child(i)->childCount()>0) item->child(i)->setExpanded(false); 
        }
        expandToDepth(item->child(i), depth-1);
      }
    }
    void expandToDepth1() { expandToDepth(objectList->invisibleRootItem(), 0); }
    void expandToDepth2() { expandToDepth(objectList->invisibleRootItem(), 1); }
    void expandToDepth3() { expandToDepth(objectList->invisibleRootItem(), 2); }
    void expandToDepth4() { expandToDepth(objectList->invisibleRootItem(), 3); }
    void expandToDepth5() { expandToDepth(objectList->invisibleRootItem(), 4); }
    void expandToDepth6() { expandToDepth(objectList->invisibleRootItem(), 5); }
    void expandToDepth7() { expandToDepth(objectList->invisibleRootItem(), 6); }
    void expandToDepth8() { expandToDepth(objectList->invisibleRootItem(), 7); }
    void expandToDepth9() { expandToDepth(objectList->invisibleRootItem(), 8); }

    void toggleEngDrawingViewSlot();
    void setOutLineAndShilouetteEdgeRecursive(QTreeWidgetItem *obj, bool enableOutLine, bool enableShilouetteEdge);
    void complexityType();
    void complexityValue();
    void loadFinished();
    void editFinishedSlot();
    void frameMinMaxSetValue(int,int);
    void selectionChanged();
  public slots:
    /** highlight the given object by enbled the bbox of this one and disabling the bbox of all others */
    void highlightObject(Object *current);
    /** highlight the given object by enbled the bbox of this one and disabling the bbox of all others */
    DLL_PUBLIC void highlightObject(std::string curID);
  public:
    DLL_PUBLIC MainWindow(std::list<std::string>& arg);
    DLL_PUBLIC ~MainWindow();
    DLL_PUBLIC bool openFile(std::string fileName, QTreeWidgetItem* parentItem=NULL, SoGroup *soParent=NULL, int ind=-1);
    void updateScene() { glViewer->getSceneManager()->render(); }
    DLL_PUBLIC static MainWindow* const getInstance();
    bool soQtEventCB(const SoEvent *const event);
    static void frameSensorCB(void *data, SoSensor*);
    void fpsCB();
    SoSepNoPickNoBBox *getSceneRootBBox() { return sceneRootBBox; }
    QTripleSlider *getTimeSlider() { return timeSlider; }
    double &getDeltaTime() { return deltaTime; }
    double getSpeed() { return speedSB->value(); }
    SoSFUInt32 *getFrame() { return frame; }
    void setTime(double t) { timeString->string.setValue(QString("Time: %2").arg(t,0,'f',5).toStdString().c_str()); }
    SoText2 *getTimeString() { return timeString; }
    SoMFColor *getBgColor() { return bgColor; }
    SoMFColor *getFgColorTop() { return fgColorTop; }
    SoMFColor *getFgColorBottom() { return fgColorBottom; }
    bool getEnableFullScreen() { return enableFullScreen; }
    void moveCameraWith(SoSFVec3f *pos, SoSFRotation *rot);
    SoDrawStyle* getOlseDrawStyle() { return olseDrawStyle; }
    SoBaseColorHeavyOverride* getOlseColor() { return olseColor; }
    SoShadowGroup* getSceneRoot() { return sceneRoot; }
    int getRootItemIndexOfChild(Group *grp) { return objectList->invisibleRootItem()->indexOfChild(grp); }
    int getReloadTimeout() { return reloadTimeout; }

    //Event for dropping
    void dragEnterEvent(QDragEnterEvent *event);
    void dropEvent(QDropEvent *event);

    DLL_PUBLIC QTreeWidget* getObjectList() { return objectList; }

  signals:
    /** This signal is emitted whenever the selected object changes.
     * Either by selecting it in the objects list or in the 3D view. */
    DLL_PUBLIC void objectSelected(std::string curID, Object *curPtr);
    /** This signal is emitted whenever a object is double clicked in the 3D view.
     * If this signal is connected to at least one slot the property dialog is no longer shown automatically. */
    DLL_PUBLIC void objectDoubleClicked(std::string curID, Object *curPtr);
    /** This signal is emmited whenever a file has been reloaded */
    DLL_PUBLIC void fileReloaded();
};

}

#endif
