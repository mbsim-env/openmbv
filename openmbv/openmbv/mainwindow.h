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

#ifndef _OPENMBVGUI_MAINWINDOW_H_
#define _OPENMBVGUI_MAINWINDOW_H_

#include "abstractviewfilter.h"
#include <Inventor/nodes/SoAnnotation.h>
#include <QMainWindow>
#include <QTreeWidget>
#include <QTextEdit>
#include <QSpinBox>
#include <QActionGroup>
#include <QLabel>
#include <QStatusBar>
#include <QtCore/QTimer>
#include <QtCore/QTime>
#include <QElapsedTimer>
#include <string>
#include <mutex>
#include "body.h"
#include "group.h"
#include "SoSpecial.h"
#include <Inventor/C/errors/debugerror.h> // workaround a include order bug in Coin-3.1.3
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/engines/SoTransformVec3f.h>
#include "SoTransposeEngine.h"
#include <Inventor/fields/SoSFRotation.h>
#include <Inventor/fields/SoSFUInt32.h>
#include <Inventor/SoOffscreenRenderer.h>
#include <Inventor/nodes/SoAsciiText.h>
#include "SoQtMyViewer.h"
#include "QTripleSlider.h"
#include <QDropEvent>
#include <qwt_wheel.h>

// If Coin and SoQt is linked as a dll no symbols of this file are exported (for an unknown reason).
// Hence we explicitly export ALL symbols.
// We cannot export selectively symbols since some symbols defined by Q_OBJECT needs also to be exported.
// The clear Qt way would be to use PImpl but this is not done here.
#ifdef _WIN32
#  define DLL_PUBLIC __declspec(dllexport)
#else
#  define DLL_PUBLIC
#endif

class QListWidgetItem;
class SoCalculator;

namespace OpenMBVGUI {
 
class MyTouchWidget;

class DialogStereo : public QDialog {
  public:
    DialogStereo();
    ~DialogStereo() override;
    void closeEvent(QCloseEvent *event) override;
    void showEvent(QShowEvent *event) override;
    MyTouchWidget *glViewerWGRight { nullptr };
  private:
    QPushButton *fullScreenButton;
};

class DLL_PUBLIC MainWindow : public QMainWindow, virtual public fmatvec::Atom {
  Q_OBJECT
  friend class DialogStereo;
  friend class Body;
  friend class Editor;
  friend class Group;
  friend class CompoundRigidBody;
  friend class IvScreenAnnotation;
  friend class MyTouchWidget;
  friend class SettingsDialog;
  private:
    static MainWindow *instance;
    enum ViewSide { top, bottom, front, back, right, left, isometric, dimetric,
                    rotateXpWorld, rotateXmWorld, rotateYpWorld, rotateYmWorld, rotateZpWorld, rotateZmWorld,
                    rotateXpScreen, rotateXmScreen, rotateYpScreen, rotateYmScreen, rotateZpScreen, rotateZmScreen };
    enum Animation { stop, play, lastFrame };
    struct WindowState { bool hasMenuBar, hasStatusBar, hasFrameSlider; };
    SoAsciiText *timeString;
    double fpsMax;
    bool enableFullScreen;
    SoTransformVec3f *cameraPosition;
    SoTransposeEngine *cameraOrientation;
    SoSwitch *worldFrameSwitch;
    SoSwitch *engDrawing;
    SoMFColor *engDrawingBGColorSaved, *engDrawingFGColorBottomSaved, *engDrawingFGColorTopSaved;
    SoFieldSensor *frameSensor;
    std::mutex mutex; // this mutex is temporarily locked during openFile calls
    bool skipWindowState{false};
    QGridLayout *mainLO;
    SoSwitch *cursorSwitch;
    SoTranslation *cursorPos;
    SoCalculator *cursorScaleE;
    SoSFFloat *mouseCursorSizeField;
    QPushButton *disableStereo;
  protected:
    SoSepNoPick *sceneRootBBox;
    QTreeWidget *objectList;
    AbstractViewFilter *objectListFilter;
    QTextEdit *objectInfo;
    QSpinBox *frameSB, *frameMinSB, *frameMaxSB;
    SoQtMyViewer *glViewer;
    SoQtMyViewer *glViewerRight { nullptr };
    DialogStereo *dialogStereo { nullptr };
    void viewChange(ViewSide side);
    SoSeparator *sceneRoot;
    SoSeparator *screenAnnotationList;
    SoScale *screenAnnotationScale1To1;
    QTimer *animTimer;
    QTimer *hdf5RefreshTimer;
    QElapsedTimer *time;
    QDoubleSpinBox *speedSB;
    int animStartFrame;
    QActionGroup *animGroup;
    QTripleSlider *timeSlider;
    double deltaTime;
    SoSFUInt32 *frame;
    QLabel *fps;
    QElapsedTimer *fpsTime;
    QwtWheel *speedWheel;
    double oldSpeed;
    QAction *stopAct, *lastFrameAct, *playAct, *toggleMenuBar, *toggleStatusBar, *toggleFrameSlider, *toggleFullScreen, *toggleDecoration, *cameraAct;
    std::shared_ptr<OpenMBV::Body> openMBVBodyForLastFrame;
    QAction *engDrawingView;
    QAction *viewStereo;

    QTimer *shortAniTimer;
    std::unique_ptr<QElapsedTimer> shortAniElapsed;
    int shortAniLast { 0 };
    void shortAni();
    std::function<void(double)> shortAniFunc;

    static void toggleAction(Object *current, QAction *currentAct);
    void execPropertyMenu(const std::vector<QAction*> &additionalActions={});
    void closeEvent(QCloseEvent *event) override;
    void showEvent(QShowEvent *event) override;
    int hdf5RefreshDelta;
    QToolBar* sceneViewToolBar;
    QMenu* sceneViewMenu;
  protected:
    void objectListClicked();
    void openFileDialog();
    void newFileDialog();
    void aboutOpenMBV();
    void guiHelp();
    void xmlHelp();
    void updateFrame(int frame_) { frame->setValue(frame_); }
    void releaseCameraFromBodySlot();
    void showWorldFrameSlot();

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
    void hdf5RefreshSlot();
    void requestHDF5Flush();
    void restartPlay();
  protected Q_SLOTS:
    void speedWheelChangedD(double value) { speedWheelChanged((int)value); }
    void speedWheelChanged(int value);
    void speedWheelPressed();
    void speedWheelReleased();
  protected:
    SoOffscreenRenderer *offScreenRenderer;
    bool exportAsPNG(short width, short height, const std::string& fileName, bool transparent);
    void exportCurrentAsPNG();
    void exportSequenceAsPNG(bool video);
    void exportCurrentAsIV();
    void exportCurrentAsPS();
    void stopSCSlot();
    void lastFrameSCSlot();
    void playSCSlot();
    void speedUpSlot();
    void speedDownSlot();
    void loadWindowState();
    void loadWindowState(std::string filename);
    void saveWindowState();
    void loadCamera();
    void loadCamera(std::string filename);
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
    void editFinishedSlot();
    void frameMinMaxSetValue(int,int);
    void selectionChanged();
    float nearPlaneValue;
    bool backgroundNeeded { true };
  public:
    SoDrawStyle *olseDrawStyle;
    SoBaseColorHeavyOverride *olseColor;
    SoDrawStyle *bboxDrawStyle;
    SoBaseColor *bboxColor;
    SoDrawStyle *highlightDrawStyle;
    SoBaseColor *highlightColor;
    SoComplexity *complexity;
    SoMFColor *bgColor, *fgColorTop, *fgColorBottom;
    MyTouchWidget *glViewerWG;
    /** highlight the given object, de-highlight all others */
    void highlightObject(Object *current);
    /** highlight ALL object with ID, de-highlight all others */
    void highlightObject(const std::string &curID);
    MainWindow(std::list<std::string>& arg, bool _skipWindowState=false);
    ~MainWindow() override;
    bool openFile(const std::string& fileName, QTreeWidgetItem* parentItem=nullptr, SoGroup *soParent=nullptr, int ind=-1);
    void updateScene() { frame->touch(); }
    static MainWindow* const getInstance();
    static void frameSensorCB(void *data, SoSensor*);
    void fpsCB();
    SoSepNoPick *getSceneRootBBox() { return sceneRootBBox; }
    QTripleSlider *getTimeSlider() { return timeSlider; }
    double &getDeltaTime() { return deltaTime; }
    double getSpeed() { return speedSB->value(); }
    SoSFUInt32 *getFrame() { return frame; }
    void setTime(double t) { timeString->string.setValue(QString("Time: %2").arg(t,0,'f',5).toStdString().c_str()); }
    SoAsciiText *getTimeString() { return timeString; }
    SoMFColor *getBgColor() { return bgColor; }
    SoMFColor *getFgColorTop() { return fgColorTop; }
    SoMFColor *getFgColorBottom() { return fgColorBottom; }
    bool getEnableFullScreen() { return enableFullScreen; }
    void moveCameraWith(SoSFVec3f *pos, SoSFRotation *rot);
    SoDrawStyle* getOlseDrawStyle() { return olseDrawStyle; }
    SoBaseColorHeavyOverride* getOlseColor() { return olseColor; }
    SoSeparator* getSceneRoot() { return sceneRoot; }
    SoSeparator* getScreenAnnotationList() { return screenAnnotationList; }
    SoScale* getScreenAnnotationScale1To1() { return screenAnnotationScale1To1; }
    int getRootItemIndexOfChild(Group *grp) { return objectList->invisibleRootItem()->indexOfChild(grp); }
    void startShortAni(const std::function<void(double)> &func, bool noAni=false);
    void setHDF5RefreshDelta(int d) { hdf5RefreshDelta=d; }

    enum class StereoType { None, LeftRight };
    void reinit3DView(StereoType stereoType);
    void setStereoOffset(double value);
    void setCameraType(SoType type);
    void setCursorPos(const SbVec3f *pos=nullptr);

    //Event for dropping
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dropEvent(QDropEvent *event) override;

    QTreeWidget* getObjectList() { return objectList; }

    std::set<void*> waitFor;
    void setNearPlaneValue(float value);
    float getNearPlaneValue() { return nearPlaneValue; }
    SoSFFloat *relCursorZ;

    QToolBar* getSceneViewToolBar() { return sceneViewToolBar; }
    QMenu* getSceneViewMenu() { return sceneViewMenu; }
    void viewAllSlot() { glViewer->viewAll(); }
    void showSettingsDialog();

    // update the flag backgroundNeeded which defines if the color gradient background is needed or not (its not needed if a VRMLBackground element exists)
    void updateBackgroundNeeded();
    bool getBackgroundNeeded() { return backgroundNeeded; }

  Q_SIGNALS:
    /** This signal is emitted whenever the selected object changes.
     * Either by selecting it in the objects list or in the 3D view.
     * (curPtr may be null if nothing is selected, curID is "" in this case) */
    void objectSelected(std::string curID, OpenMBVGUI::Object *curPtr); // The OpenMBVGUI namespace prefix is needed to enable Qt signal handling with SIGNAL(...)

    /** This signal is emitted whenever a object is double clicked in the 3D view.
     * If this signal is connected to at least one slot the property dialog is no longer shown automatically. */
    void objectDoubleClicked(std::string curID, OpenMBVGUI::Object *curPtr); // The OpenMBVGUI namespace prefix is needed to enable Qt signal handling with SIGNAL(...)

    /** This signal is emitted after a reload of a Group */
    void fileReloaded(OpenMBVGUI::Group *grp); // The OpenMBVGUI namespace prefix is needed to enable Qt signal handling with SIGNAL(...)
};

}

#endif
