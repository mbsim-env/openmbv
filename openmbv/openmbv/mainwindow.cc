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
#include <openmbvcppinterface/group.h>
#include <openmbvcppinterface/cube.h>
#include <openmbvcppinterface/compoundrigidbody.h>
#include "mainwindow.h"
#include <algorithm>
#include <Inventor/Qt/SoQt.h>
#include <QDesktopWidget>
#include <QDesktopServices>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QInputDialog>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QFileDialog>
#include <QtGui/QMouseEvent>
#include <QtWidgets/QApplication>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QColorDialog>
#include <QtCore/QElapsedTimer>
#include <QShortcut>
#include <QMimeData>
#include <QSettings>
#include <QMetaMethod>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoCone.h>
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <Inventor/nodes/SoRotation.h>
#include <Inventor/nodes/SoDirectionalLight.h>
#include <Inventor/nodes/SoEventCallback.h>
#include <Inventor/nodes/SoLightModel.h>
#include <Inventor/nodes/SoDepthBuffer.h>
#include <Inventor/nodes/SoPolygonOffset.h>
#include <Inventor/nodes/SoShapeHints.h>
#include <Inventor/events/SoMouseButtonEvent.h>
#include <Inventor/events/SoLocation2Event.h>
#include <Inventor/sensors/SoFieldSensor.h>
#include <Inventor/actions/SoWriteAction.h>
#include <Inventor/nodes/SoComplexity.h>
#include <Inventor/annex/HardCopy/SoVectorizePSAction.h>
#include "exportdialog.h"
#include "object.h"
#include "cuboid.h"
#include "group.h"
#include "objectfactory.h"
#include "compoundrigidbody.h"
#include <string>
#include <set>
#include <hdf5serie/file.h>
#include <Inventor/SbViewportRegion.h>
#include <Inventor/actions/SoRayPickAction.h>
#include <Inventor/SoPickedPoint.h>
#include "IndexedTesselationFace.h"
#include "utils.h"
#include <boost/dll.hpp>

using namespace std;
using namespace std::placeholders;

namespace OpenMBVGUI {

enum Mode { no, rotate, translate, zoom };

MainWindow *MainWindow::instance=nullptr;

QObject* qTreeWidgetItemToQObject(const QModelIndex &index) {
  return dynamic_cast<QObject*>(static_cast<QTreeWidgetItem*>(index.internalPointer()));
}

MainWindow::MainWindow(list<string>& arg) :  fpsMax(25), enableFullScreen(false), deltaTime(0), oldSpeed(1) {
  boost::filesystem::path installPath(boost::dll::program_location().parent_path().parent_path());
  // If <local>/lib/dri exists use it as load path for GL DRI drivers.
  // DRI drivers depend on libstdc++.so. Hence, they must be distributed with the binary distribution.
  if(boost::filesystem::exists(installPath/"lib"/"dri")) {
    static char DRI[2048];
    putenv(strcat(strcpy(DRI, "LIBGL_DRIVERS_PATH="), (installPath/"lib"/"dri").string().c_str()));
  }

  if(instance) throw runtime_error("The class MainWindow is a singleton class!");
  instance=this;

  list<string>::iterator i, i2;

  setWindowTitle("OpenMBV - Open Multi Body Viewer");
  setWindowIcon(Utils::QIconCached((installPath/"share"/"openmbv"/"icons"/"openmbv.svg").string()));

  // init Utils
  Utils::initialize();

  // init SoQt and Inventor
  SoQt::init(this);
  // init user engines
  SoTransposeEngine::initClass();
  IndexedTesselationFace::initClass();
  // init realtime
  SoDB::enableRealTimeSensor(false);
  SoSceneManager::enableRealTimeUpdate(false);

  // initialize global frame field
  frame=(SoSFUInt32*)SoDB::createGlobalField("frame",SoSFUInt32::getClassTypeId());
  frame->getContainer()->ref(); // reference the global field
  frame->setValue(0);

  engDrawingBGColorSaved=new SoMFColor();
  engDrawingFGColorBottomSaved=new SoMFColor();
  engDrawingFGColorTopSaved=new SoMFColor();

  // main widget
  QWidget *mainWG=new QWidget(this);
  setCentralWidget(mainWG);
  auto *mainLO=new QGridLayout(mainWG);
  mainLO->setContentsMargins(0,0,0,0);
  mainWG->setLayout(mainLO);
  // gl viewer
  QWidget *glViewerWG=new QWidget(this);
  timeString=new SoText2;
  timeString->ref();
  bgColor=new SoMFColor;
  bgColor->set1Value(0, 0.35,0.35,0.6);
  bgColor->set1Value(1, 0.35,0.35,0.6);
  bgColor->set1Value(2, 0.83,0.83,1.0);
  bgColor->set1Value(3, 0.83,0.83,1.0);
  fgColorTop=new SoMFColor;
  fgColorTop->set1Value(0, 0,0,0);
  fgColorBottom=new SoMFColor;
  fgColorBottom->set1Value(0, 1,1,1);
  int transparency=1;
  if((i=std::find(arg.begin(), arg.end(), "--transparency"))!=arg.end()) {
    i2=i; i2++;
    transparency=QString(i2->c_str()).toDouble();
    arg.erase(i); arg.erase(i2);
  }
  glViewer=new SoQtMyViewer(glViewerWG, transparency);
  mainLO->addWidget(glViewerWG,0,0);
  sceneRoot=new SoSeparator;
  sceneRoot->ref();
  auto *offset=new SoPolygonOffset; // move all filled polygons in background
  sceneRoot->addChild(offset);
  offset->factor.setValue(0.0);
  offset->units.setValue(1000);
  complexity=new SoComplexity;
  sceneRoot->addChild(complexity);
  // enable backface culling (and one sided lightning) by default
  auto *sh=new SoShapeHints;
  sceneRoot->addChild(sh);
  sh->vertexOrdering=SoShapeHints::COUNTERCLOCKWISE;
  sh->shapeType=SoShapeHints::SOLID;

  // Switch for global shilouette/crease/boundary override elements
  engDrawing=new SoSwitch;
  sceneRoot->addChild(engDrawing);
  engDrawing->whichChild.setValue(SO_SWITCH_NONE);
  auto *lm3=new SoLightModel;
  lm3->model.setValue(SoLightModel::BASE_COLOR);
  lm3->setOverride(true);
  engDrawing->addChild(lm3);
  auto *bc=new SoBaseColor;
  bc->rgb.setValue(1,1,1);
  bc->setOverride(true);
  engDrawing->addChild(bc);

  sceneRootBBox=new SoSepNoPick;
  sceneRoot->addChild(sceneRootBBox);
  auto *lm=new SoLightModel;
  sceneRootBBox->addChild(lm);
  lm->model.setValue(SoLightModel::BASE_COLOR);
  auto *color=new SoBaseColor;
  sceneRootBBox->addChild(color);
  color->rgb.setValue(0,1,0);
  auto *style=new SoDrawStyle;
  style->style.setValue(SoDrawStyle::LINES);
  style->lineWidth.setValue(2);
  sceneRootBBox->addChild(style);

  // Move the world system such that the camera is constant relative the the body
  // with moves with the camera; if not, don't move the world system.
  // rot
  auto *worldSysRot=new SoRotation;
  sceneRoot->addChild(worldSysRot);
  cameraOrientation=new SoTransposeEngine;
  cameraOrientation->ref();
  worldSysRot->rotation.connectFrom(&cameraOrientation->outRotation);
  // trans
  auto *worldSysTrans=new SoTranslation;
  sceneRoot->addChild(worldSysTrans);
  cameraPosition=new SoTransformVec3f;
  cameraPosition->ref();
  cameraPosition->matrix.setValue(-1,0,0,0 , 0,-1,0,0 , 0,0,-1,0 , 0,0,0,1);
  worldSysTrans->translation.connectFrom(&cameraPosition->point);

  // world frame
  worldFrameSwitch=new SoSwitch;
  sceneRoot->addChild(worldFrameSwitch);
  worldFrameSwitch->whichChild.setValue(SO_SWITCH_NONE);
  auto *worldFrameSep=new SoSepNoPickNoBBox;
  worldFrameSwitch->addChild(worldFrameSep);
  auto *drawStyle=new SoDrawStyle;
  worldFrameSep->addChild(drawStyle);
  drawStyle->lineWidth.setValue(2);
  drawStyle->linePattern.setValue(0xF8F8);
  worldFrameSep->addChild(Utils::soFrame(1,1,false));
  glViewer->setSceneGraph(sceneRoot);
  
  // time slider
  timeSlider=new QTripleSlider(this);
  mainLO->addWidget(timeSlider, 0, 1);
  timeSlider->setTotalRange(0, 0);
  connect(timeSlider, &QTripleSlider::sliderMoved, this, &MainWindow::updateFrame);

  // object list dock widget
  QDockWidget *objectListDW=new QDockWidget(tr("Objects"),this);
  objectListDW->setObjectName("MainWindow::objectListDW");
  QWidget *objectListWG=new QWidget(this);
  auto *objectListLO=new QGridLayout(objectListWG);
  objectListWG->setLayout(objectListLO);
  objectListDW->setWidget(objectListWG);
  addDockWidget(Qt::LeftDockWidgetArea,objectListDW);
  objectList = new QTreeWidget(objectListDW);
  objectListFilter=new AbstractViewFilter(objectList, 0, -1, "OpenMBVGUI::", &qTreeWidgetItemToQObject);
  objectListLO->addWidget(objectListFilter, 0,0);
  objectListLO->addWidget(objectList, 1,0);
  objectList->setHeaderHidden(true);
  objectList->setSelectionMode(QAbstractItemView::ExtendedSelection);
  connect(objectList,&QTreeWidget::pressed, this, &MainWindow::objectListClicked);
  connect(objectList,&QTreeWidget::itemCollapsed, this, &MainWindow::collapseItem);
  connect(objectList,&QTreeWidget::itemExpanded, this, &MainWindow::expandItem);
  connect(objectList,&QTreeWidget::itemSelectionChanged, this, &MainWindow::selectionChanged);
  connect(new QShortcut(QKeySequence("1"),this), &QShortcut::activated, this, &MainWindow::expandToDepth1);
  connect(new QShortcut(QKeySequence("2"),this), &QShortcut::activated, this, &MainWindow::expandToDepth2);
  connect(new QShortcut(QKeySequence("3"),this), &QShortcut::activated, this, &MainWindow::expandToDepth3);
  connect(new QShortcut(QKeySequence("4"),this), &QShortcut::activated, this, &MainWindow::expandToDepth4);
  connect(new QShortcut(QKeySequence("5"),this), &QShortcut::activated, this, &MainWindow::expandToDepth5);
  connect(new QShortcut(QKeySequence("6"),this), &QShortcut::activated, this, &MainWindow::expandToDepth6);
  connect(new QShortcut(QKeySequence("7"),this), &QShortcut::activated, this, &MainWindow::expandToDepth7);
  connect(new QShortcut(QKeySequence("8"),this), &QShortcut::activated, this, &MainWindow::expandToDepth8);
  connect(new QShortcut(QKeySequence("9"),this), &QShortcut::activated, this, &MainWindow::expandToDepth9);
  objectList->setEditTriggers(QTreeWidget::EditKeyPressed);
  connect(objectList->itemDelegate(), &QAbstractItemDelegate::closeEditor, this, &MainWindow::editFinishedSlot);

  // object info dock widget
  QDockWidget *objectInfoDW=new QDockWidget(tr("Object Info"),this);
  objectInfoDW->setObjectName("MainWindow::objectInfoDW");
  QWidget *objectInfoWG=new QWidget;
  auto *objectInfoLO=new QGridLayout;
  objectInfoWG->setLayout(objectInfoLO);
  objectInfoDW->setWidget(objectInfoWG);
  addDockWidget(Qt::LeftDockWidgetArea,objectInfoDW);
  objectInfo = new QTextEdit(objectInfoDW);
  objectInfoLO->addWidget(objectInfo, 0,0);
  objectInfo->setReadOnly(true);
  objectInfo->setLineWrapMode(QTextEdit::NoWrap);
  connect(objectList,&QTreeWidget::currentItemChanged,this,&MainWindow::setObjectInfo);

  // menu bar
  auto *mb=new QMenuBar(this);
  setMenuBar(mb);

  QAction *act;
  // file menu
  QMenu *fileMenu=new QMenu("File", menuBar());
  QAction *addFileAct=fileMenu->addAction(Utils::QIconCached("addfile.svg"), "Add file...", this, &MainWindow::openFileDialog);
  fileMenu->addAction(Utils::QIconCached("newfile.svg"), "New file...", this, &MainWindow::newFileDialog);
  fileMenu->addSeparator();
  act=fileMenu->addAction(Utils::QIconCached("exportimg.svg"), "Export current frame as PNG...", this, &MainWindow::exportCurrentAsPNG, QKeySequence("Ctrl+P"));
  addAction(act); // must work also if menu bar is invisible
  act=fileMenu->addAction(Utils::QIconCached("exportimgsequence.svg"), "Export frame sequence as PNG...", this, &MainWindow::exportSequenceAsPNG, QKeySequence("Ctrl+Shift+P"));
  addAction(act); // must work also if menu bar is invisible
  fileMenu->addAction(Utils::QIconCached("exportiv.svg"), "Export current frame as IV...", this, &MainWindow::exportCurrentAsIV);
  fileMenu->addAction(Utils::QIconCached("exportiv.svg"), "Export current frame as PS...", this, &MainWindow::exportCurrentAsPS);
  fileMenu->addSeparator();
  fileMenu->addAction(Utils::QIconCached("loadwst.svg"), "Load window state...", this, static_cast<void(MainWindow::*)()>(&MainWindow::loadWindowState));
  act=fileMenu->addAction(Utils::QIconCached("savewst.svg"), "Save window state...", this, &MainWindow::saveWindowState, QKeySequence("Ctrl+W"));
  addAction(act); // must work also if menu bar is invisible
  fileMenu->addAction(Utils::QIconCached("loadcamera.svg"), "Load camera...", this, static_cast<void(MainWindow::*)()>(&MainWindow::loadCamera));
  act=fileMenu->addAction(Utils::QIconCached("savecamera.svg"), "Save camera...", this, &MainWindow::saveCamera, QKeySequence("Ctrl+C"));
  act->setShortcutContext(Qt::WidgetWithChildrenShortcut);
  addAction(act); // must work also if menu bar is invisible
  fileMenu->addSeparator();
  act=fileMenu->addAction(Utils::QIconCached("quit.svg"), "Exit", qApp, &QApplication::quit);
  addAction(act); // must work also if menu bar is invisible
  menuBar()->addMenu(fileMenu);

  // animation menu
  stopAct=new QAction(Utils::QIconCached("stop.svg"), "Stop", this);
  addAction(stopAct); // must work also if menu bar is invisible
  stopAct->setShortcut(QKeySequence("S"));
  lastFrameAct=new QAction(Utils::QIconCached("lastframe.svg"), "Last frame", this);
  addAction(lastFrameAct); // must work also if menu bar is invisible
  lastFrameAct->setShortcut(QKeySequence("L"));
  playAct=new QAction(Utils::QIconCached("play.svg"),           "Play", this);
  addAction(playAct); // must work also if menu bar is invisible
  playAct->setShortcut(QKeySequence("P"));
  stopAct->setCheckable(true);
  stopAct->setData(QVariant(stop));
  playAct->setCheckable(true);
  playAct->setData(QVariant(play));
  lastFrameAct->setCheckable(true);
  lastFrameAct->setData(QVariant(lastFrame));
  stopAct->setChecked(true);
  QMenu *animationMenu=new QMenu("Animation", menuBar());
  animationMenu->addAction(stopAct);
  animationMenu->addAction(lastFrameAct);
  animationMenu->addAction(playAct);
  menuBar()->addMenu(animationMenu);
  connect(stopAct, &QAction::triggered, this, &MainWindow::stopSCSlot);
  connect(lastFrameAct, &QAction::triggered, this, &MainWindow::lastFrameSCSlot);
  connect(playAct, &QAction::triggered, this, &MainWindow::playSCSlot);


  // scene view menu
  QMenu *sceneViewMenu=new QMenu("Scene View", menuBar());
  QAction *viewAllAct=sceneViewMenu->addAction(Utils::QIconCached("viewall.svg"),"View all", this, &MainWindow::viewAllSlot, QKeySequence("A"));
  addAction(viewAllAct); // must work also if menu bar is invisible
  QMenu *axialView=sceneViewMenu->addMenu(Utils::QIconCached("axialview.svg"),"Axial view");
  QAction *topViewAct=axialView->addAction(Utils::QIconCached("topview.svg"),"Top", this, &MainWindow::viewTopSlot, QKeySequence("T"));
  addAction(topViewAct); // must work also if menu bar is invisible
  QAction *bottomViewAct=axialView->addAction(Utils::QIconCached("bottomview.svg"),"Bottom", this, &MainWindow::viewBottomSlot, QKeySequence("Shift+T"));
  addAction(bottomViewAct); // must work also if menu bar is invisible
  QAction *frontViewAct=axialView->addAction(Utils::QIconCached("frontview.svg"),"Front", this, &MainWindow::viewFrontSlot, QKeySequence("F"));
  addAction(frontViewAct); // must work also if menu bar is invisible
  QAction *backViewAct=axialView->addAction(Utils::QIconCached("backview.svg"),"Back", this, &MainWindow::viewBackSlot, QKeySequence("Shift+F"));
  addAction(backViewAct); // must work also if menu bar is invisible
  QAction *rightViewAct=axialView->addAction(Utils::QIconCached("rightview.svg"),"Right", this, &MainWindow::viewRightSlot, QKeySequence("R"));
  addAction(rightViewAct); // must work also if menu bar is invisible
  QAction *leftViewAct=axialView->addAction(Utils::QIconCached("leftview.svg"),"Left", this, &MainWindow::viewLeftSlot, QKeySequence("Shift+R"));
  addAction(leftViewAct); // must work also if menu bar is invisible
  QMenu *spaceView=sceneViewMenu->addMenu(Utils::QIconCached("spaceview.svg"),"Space view");
  QAction *isometriViewAct=spaceView->addAction(Utils::QIconCached("isometricview.svg"),"Isometric", this, &MainWindow::viewIsometricSlot);
  QAction *dimetricViewAct=spaceView->addAction(Utils::QIconCached("dimetricview.svg"),"Dimetric", this, &MainWindow::viewDimetricSlot);
  // QKeySequence("D") is used by SoQtMyViewer for dragger manipulation
  QMenu *rotateView=sceneViewMenu->addMenu(Utils::QIconCached("rotateview.svg"),"Rotate view");
  act=rotateView->addAction("+10deg About World-X-Axis", this, &MainWindow::viewRotateXpWorld, QKeySequence("X"));
  addAction(act);
  act=rotateView->addAction("-10deg About World-X-Axis", this, &MainWindow::viewRotateXmWorld, QKeySequence("Shift+X"));
  addAction(act);
  act=rotateView->addAction("+10deg About World-Y-Axis", this, &MainWindow::viewRotateYpWorld, QKeySequence("Y"));
  addAction(act);
  act=rotateView->addAction("-10deg About World-Y-Axis", this, &MainWindow::viewRotateYmWorld, QKeySequence("Shift+Y"));
  addAction(act);
  act=rotateView->addAction("+10deg About World-Z-Axis", this, &MainWindow::viewRotateZpWorld, QKeySequence("Z"));
  addAction(act);
  act=rotateView->addAction("-10deg About World-Z-Axis", this, &MainWindow::viewRotateZmWorld, QKeySequence("Shift+Z"));
  addAction(act);
  rotateView->addSeparator();
  act=rotateView->addAction("+10deg About Screen-X-Axis", this, &MainWindow::viewRotateXpScreen, QKeySequence("Ctrl+X"));
  act->setShortcutContext(Qt::WidgetWithChildrenShortcut);
  addAction(act);
  act=rotateView->addAction("-10deg About Screen-X-Axis", this, &MainWindow::viewRotateXmScreen, QKeySequence("Ctrl+Shift+X"));
  addAction(act);
  act=rotateView->addAction("+10deg About Screen-Y-Axis", this, &MainWindow::viewRotateYpScreen, QKeySequence("Ctrl+Y"));
  addAction(act);
  act=rotateView->addAction("-10deg About Screen-Y-Axis", this, &MainWindow::viewRotateYmScreen, QKeySequence("Ctrl+Shift+Y"));
  addAction(act);
  act=rotateView->addAction("+10deg About Screen-Z-Axis", this, &MainWindow::viewRotateZpScreen, QKeySequence("Ctrl+Z"));
  act->setShortcutContext(Qt::WidgetWithChildrenShortcut);
  addAction(act);
  act=rotateView->addAction("-10deg About Screen-Z-Axis", this, &MainWindow::viewRotateZmScreen, QKeySequence("Ctrl+Shift+Z"));
  act->setShortcutContext(Qt::WidgetWithChildrenShortcut);
  addAction(act);
  sceneViewMenu->addSeparator();
  act=sceneViewMenu->addAction(Utils::QIconCached("frame.svg"),"World frame", this, &MainWindow::showWorldFrameSlot, QKeySequence("W"));
  act->setCheckable(true);
  sceneViewMenu->addAction(Utils::QIconCached("olselinewidth.svg"),"Outline and shilouette edge line width...", this, &MainWindow::olseLineWidthSlot);
  sceneViewMenu->addAction(Utils::QIconCached("olsecolor.svg"),"Outline and shilouette edge color...", this, &MainWindow::olseColorSlot);
  sceneViewMenu->addAction(Utils::QIconCached("complexitytype.svg"),"Complexity type...", this, &MainWindow::complexityType);
  sceneViewMenu->addAction(Utils::QIconCached("complexityvalue.svg"),"Complexity value...", this, &MainWindow::complexityValue);
  sceneViewMenu->addSeparator();
  QAction *cameraAct=sceneViewMenu->addAction(Utils::QIconCached("camera.svg"),"Toggle camera type", this, &MainWindow::toggleCameraTypeSlot, QKeySequence("C"));
  addAction(cameraAct); // must work also if menu bar is invisible
  sceneViewMenu->addAction(Utils::QIconCached("camerabody.svg"),"Release camera from move with body", this, &MainWindow::releaseCameraFromBodySlot);
  sceneViewMenu->addSeparator();
  engDrawingView=sceneViewMenu->addAction(Utils::QIconCached("engdrawing.svg"),"Engineering drawing", this, &MainWindow::toggleEngDrawingViewSlot);
  engDrawingView->setToolTip("NOTE: If getting unchecked, the outlines of all bodies will be enabled and the shilouette edges are disabled!");
  engDrawingView->setStatusTip(engDrawingView->toolTip());
  engDrawingView->setCheckable(true);
  topBGColorAct=sceneViewMenu->addAction(Utils::QIconCached("bgcolor.svg"),"Top background color...", this, &MainWindow::topBGColor);
  bottomBGColorAct=sceneViewMenu->addAction(Utils::QIconCached("bgcolor.svg"),"Bottom background color...", this, &MainWindow::bottomBGColor);
  menuBar()->addMenu(sceneViewMenu);

  // gui view menu
  QMenu *guiViewMenu=new QMenu("GUI View", menuBar());
  toggleMenuBar=guiViewMenu->addAction("Menu bar", this, &MainWindow::toggleMenuBarSlot, QKeySequence("F10"));
  addAction(toggleMenuBar); // must work also if menu bar is invisible
  toggleMenuBar->setCheckable(true);
  toggleMenuBar->setChecked(true);
  toggleStatusBar=guiViewMenu->addAction("Status bar", this, &MainWindow::toggleStatusBarSlot);
  toggleStatusBar->setCheckable(true);
  toggleStatusBar->setChecked(true);
  toggleFrameSlider=guiViewMenu->addAction("Frame/Time slider", this, &MainWindow::toggleFrameSliderSlot);
  toggleFrameSlider->setCheckable(true);
  toggleFrameSlider->setChecked(true);
  QAction *toggleFullScreen=guiViewMenu->addAction("Full screen", this, &MainWindow::toggleFullScreenSlot, QKeySequence("F5"));
  addAction(toggleFullScreen); // must work also if menu bar is invisible
  toggleFullScreen->setCheckable(true);
  toggleDecoration=guiViewMenu->addAction("Window decoration", this, &MainWindow::toggleDecorationSlot);
  toggleDecoration->setCheckable(true);
  toggleDecoration->setChecked(true);
  menuBar()->addMenu(guiViewMenu);

  // dock menu
  QMenu *dockMenu=new QMenu("Docks", menuBar());
  dockMenu->addAction(objectListDW->toggleViewAction());
  dockMenu->addAction(objectInfoDW->toggleViewAction());
  menuBar()->addMenu(dockMenu);

  // file toolbar
  QToolBar *fileTB=new QToolBar("FileToolBar", this);
  fileTB->setObjectName("MainWindow::fileTB");
  addToolBar(Qt::TopToolBarArea, fileTB);
  fileTB->addAction(addFileAct);

  // view toolbar
  QToolBar *viewTB=new QToolBar("ViewToolBar", this);
  viewTB->setObjectName("MainWindow::viewTB");
  addToolBar(Qt::TopToolBarArea, viewTB);
  viewTB->addAction(viewAllAct);
  viewTB->addSeparator();
  viewTB->addAction(topViewAct);
  viewTB->addAction(bottomViewAct);
  viewTB->addAction(frontViewAct);
  viewTB->addAction(backViewAct);
  viewTB->addAction(rightViewAct);
  viewTB->addAction(leftViewAct);
  viewTB->addSeparator();
  viewTB->addAction(isometriViewAct);
  viewTB->addAction(dimetricViewAct);
  viewTB->addSeparator();
  viewTB->addAction(cameraAct);

  // animation toolbar
  QToolBar *animationTB=new QToolBar("AnimationToolBar", this);
  animationTB->setObjectName("MainWindow::animationTB");
  addToolBar(Qt::TopToolBarArea, animationTB);
  // stop button
  animationTB->addAction(stopAct);
  // last frame button
  animationTB->addAction(lastFrameAct);
  // play button
  animationTB->addAction(playAct);
  // separator
  animationTB->addSeparator();
  // speed spin box
  speedSB=new QDoubleSpinBox;
  speedSB->setRange(1e-30, 1e30);
  speedSB->setMaximumSize(50, 1000);
  speedSB->setDecimals(3);
  speedSB->setButtonSymbols(QDoubleSpinBox::NoButtons);
  speedSB->setValue(1.0);
  connect(speedSB, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &MainWindow::restartPlay);
  QWidget *speedWG=new QWidget(this);
  auto *speedLO=new QGridLayout(speedWG);
  speedLO->setSpacing(0);
  speedLO->setContentsMargins(0,0,0,0);
  speedWG->setLayout(speedLO);
  QLabel *speedL=new QLabel("Speed:", this);
  speedLO->addWidget(speedL, 0, 0);
  speedLO->addWidget(speedSB, 1, 0);
  speedWheel=new QwtWheel(this);
  speedWheel->setWheelWidth(10);
  speedWheel->setTotalAngle(360*15);
  connect(speedWheel, &QwtWheel::valueChanged, this, &MainWindow::speedWheelChangedD);
  connect(speedWheel, &QwtWheel::wheelPressed, this, &MainWindow::speedWheelPressed);
  connect(speedWheel, &QwtWheel::wheelReleased, this, &MainWindow::speedWheelReleased);
  speedWheel->setRange(-20000, 20000);
  speedWheel->setOrientation(Qt::Vertical);
  speedLO->addWidget(speedWheel, 0, 1, 2, 1);
  animationTB->addWidget(speedWG);
  connect(new QShortcut(QKeySequence(Qt::Key_PageUp),this), &QShortcut::activated, this, &MainWindow::speedUpSlot);
  connect(new QShortcut(QKeySequence(Qt::Key_PageDown),this), &QShortcut::activated, this, &MainWindow::speedDownSlot);
  animationTB->addSeparator();
  // frame spin box
  frameSB=new QSpinBox;
  frameSB->setMinimumSize(55,0);
  QWidget *frameWG=new QWidget(this);
  auto *frameLO=new QGridLayout(frameWG);
  frameLO->setSpacing(0);
  frameLO->setContentsMargins(0,0,0,0);
  frameWG->setLayout(frameLO);
  QLabel *frameL=new QLabel("Frame:", this);
  frameLO->addWidget(frameL, 0, 0);
  frameLO->addWidget(frameSB, 1, 0);
  animationTB->addWidget(frameWG);
  connect(frameSB, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &MainWindow::updateFrame);
  connect(timeSlider, &QTripleSlider::currentRangeChanged, this, &MainWindow::frameSBSetRange);
  connect(timeSlider, &QTripleSlider::currentRangeChanged, this, &MainWindow::restartPlay);
  connect(timeSlider, &QTripleSlider::currentRangeChanged, this, &MainWindow::frameMinMaxSetValue);
  connect(new QShortcut(QKeySequence(Qt::Key_Right),this), &QShortcut::activated, frameSB, &QSpinBox::stepUp);
  connect(new QShortcut(QKeySequence(Qt::Key_Left),this), &QShortcut::activated, frameSB, &QSpinBox::stepDown);
  connect(new QShortcut(QKeySequence(Qt::Key_J),this), &QShortcut::activated, frameSB, &QSpinBox::stepUp);
  connect(new QShortcut(QKeySequence(Qt::Key_K),this), &QShortcut::activated, frameSB, &QSpinBox::stepDown);
  // min frame spin box
  frameMinSB=new QSpinBox;
  frameMinSB->setMinimumSize(55,0);
  frameMinSB->setRange(0, 0);
  QWidget *frameMinWG=new QWidget(this);
  auto *frameMinLO=new QGridLayout(frameMinWG);
  frameMinLO->setSpacing(0);
  frameMinLO->setContentsMargins(0,0,0,0);
  frameMinWG->setLayout(frameMinLO);
  QLabel *frameMinL=new QLabel("Min:", this);
  frameMinLO->addWidget(frameMinL, 0, 0);
  frameMinLO->addWidget(frameMinSB, 1, 0);
  animationTB->addWidget(frameMinWG);
  connect(frameMinSB, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), timeSlider, &QTripleSlider::setCurrentMinimum);
  // max frame spin box
  frameMaxSB=new QSpinBox;
  frameMaxSB->setMinimumSize(55,0);
  frameMaxSB->setRange(0, 0);
  QWidget *frameMaxWG=new QWidget(this);
  auto *frameMaxLO=new QGridLayout(frameMaxWG);
  frameMaxLO->setSpacing(0);
  frameMaxLO->setContentsMargins(0,0,0,0);
  frameMaxWG->setLayout(frameMaxLO);
  QLabel *frameMaxL=new QLabel("Max:", this);
  frameMaxLO->addWidget(frameMaxL, 0, 0);
  frameMaxLO->addWidget(frameMaxSB, 1, 0);
  animationTB->addWidget(frameMaxWG);
  connect(frameMaxSB, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), timeSlider, &QTripleSlider::setCurrentMaximum);

  // tool menu
  QMenu *toolMenu=new QMenu("Tools", menuBar());
  toolMenu->addAction(fileTB->toggleViewAction());
  toolMenu->addAction(viewTB->toggleViewAction());
  toolMenu->addAction(animationTB->toggleViewAction());
  menuBar()->addMenu(toolMenu);

  // help menu
  menuBar()->addSeparator();
  QMenu *helpMenu=new QMenu("Help", menuBar());
  helpMenu->addAction(Utils::QIconCached("help.svg"), "GUI help...", this, &MainWindow::guiHelp);
  helpMenu->addAction(Utils::QIconCached("help.svg"), "XML help...", this, &MainWindow::xmlHelp);
  helpMenu->addAction(Utils::QIconCached((installPath/"share"/"openmbv"/"icons"/"openmbv.svg").string().c_str()),
                      "About OpenMBV...", this, &MainWindow::aboutOpenMBV);
  menuBar()->addMenu(helpMenu);

  // status bar
  auto *sb=new QStatusBar(this);
  fps=new QLabel("FPS: -");
  fpsTime=new QTime();
  sb->addPermanentWidget(fps);
  setStatusBar(sb);

  // register callback function on frame change
  frameSensor=new SoFieldSensor(frameSensorCB, this);
  frameSensor->attach(frame);

  // animation timer
  animTimer=new QTimer(this);
  connect(animTimer, &QTimer::timeout, this, &MainWindow::heavyWorkSlot);
  time=new QTime();

  // react on parameters

  // line width for outline and shilouette edges
  olseDrawStyle=new SoDrawStyle;
  olseDrawStyle->ref();
  olseDrawStyle->style.setValue(SoDrawStyle::LINES);
  olseDrawStyle->lineWidth.setValue(1);
  if((i=std::find(arg.begin(), arg.end(), "--olselinewidth"))!=arg.end()) {
    i2=i; i2++;
    olseDrawStyle->lineWidth.setValue(QString(i2->c_str()).toDouble());
    arg.erase(i); arg.erase(i2);
  }

  // complexity
  complexity->type.setValue(SoComplexity::SCREEN_SPACE);
  if((i=std::find(arg.begin(), arg.end(), "--complexitytype"))!=arg.end()) {
    i2=i; i2++;
    if(*i2=="objectspace") complexity->type.setValue(SoComplexity::OBJECT_SPACE);
    if(*i2=="screenspace") complexity->type.setValue(SoComplexity::SCREEN_SPACE);
    if(*i2=="boundingbox") complexity->type.setValue(SoComplexity::BOUNDING_BOX);
    arg.erase(i); arg.erase(i2);
  }
  complexity->value.setValue(0.2);
  if((i=std::find(arg.begin(), arg.end(), "--complexityvalue"))!=arg.end()) {
    i2=i; i2++;
    complexity->value.setValue(QString(i2->c_str()).toDouble()/100);
    arg.erase(i); arg.erase(i2);
  }

  // color for outline and shilouette edges
  olseColor=new SoBaseColorHeavyOverride;
  olseColor->ref();
  olseColor->rgb.set1Value(0, 0,0,0);
  if((i=std::find(arg.begin(), arg.end(), "--olsecolor"))!=arg.end()) {
    i2=i; i2++;
    QColor color(i2->c_str());
    if(color.isValid()) {
      QRgb rgb=color.rgb();
      olseColor->rgb.set1Value(0, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
    }
    arg.erase(i); arg.erase(i2);
  }

  // background color
  if((i=std::find(arg.begin(), arg.end(), "--topbgcolor"))!=arg.end()) {
    i2=i; i2++;
    QColor color(i2->c_str());
    if(color.isValid()) {
      QRgb rgb=color.rgb();
      bgColor->set1Value(2, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
      bgColor->set1Value(3, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
      fgColorTop->set1Value(0, 1-(color.value()+127)/255,1-(color.value()+127)/255,1-(color.value()+127)/255);
    }
    arg.erase(i); arg.erase(i2);
  }
  if((i=std::find(arg.begin(), arg.end(), "--bottombgcolor"))!=arg.end()) {
    i2=i; i2++;
    QColor color(i2->c_str());
    if(color.isValid()) {
      QRgb rgb=color.rgb();
      bgColor->set1Value(0, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
      bgColor->set1Value(1, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
      fgColorBottom->set1Value(0, 1-(color.value()+127)/255,1-(color.value()+127)/255,1-(color.value()+127)/255);
    }
    arg.erase(i); arg.erase(i2);
  }

  // close all docks and toolbars
  if((i=std::find(arg.begin(), arg.end(), "--closeall"))!=arg.end()) {
    objectListDW->close();
    objectInfoDW->close();
    fileTB->close();
    viewTB->close();
    animationTB->close();
    menuBar()->setVisible(false);
    statusBar()->setVisible(false);
    timeSlider->setVisible(false);
    toggleMenuBar->setChecked(false);
    toggleStatusBar->setChecked(false);
    toggleFrameSlider->setChecked(false);
    arg.erase(i);
  }

  // fullscreen
  if((i=std::find(arg.begin(), arg.end(), "--fullscreen"))!=arg.end()) {
    enableFullScreen=true;
    toggleFullScreen->setChecked(true);
    arg.erase(i);
  }

  // decoratoin
  if((i=std::find(arg.begin(), arg.end(), "--nodecoration"))!=arg.end()) {
    toggleDecoration->setChecked(false);
    setWindowFlags(Qt::FramelessWindowHint);
    show();
    arg.erase(i);
  }

  // load the window state file
  if((i=std::find(arg.begin(), arg.end(), "--wst"))!=arg.end()) {
    i2=i; i2++;
    loadWindowState(*i2);
    arg.erase(i); arg.erase(i2);
  }

  // resize if no --wst and no --geometry option is given
  if(std::find(arg.begin(), arg.end(), "--wst")==arg.end() &&
     std::find(arg.begin(), arg.end(), "--geometry")==arg.end())
    resize(830,560);
  
  // geometry
  if((i=std::find(arg.begin(), arg.end(), "--geometry"))!=arg.end()) {
    i2=i; i2++;
    QRegExp re("^([0-9]+)x([0-9]+)(?:\\+([0-9]+))?(?:\\+([0-9]+))?$");
    re.indexIn(i2->c_str());
    resize(re.cap(1).toInt(), re.cap(2).toInt());
    bool xok, yok;
    int x=re.cap(3).toInt(&xok);
    int y=re.cap(4).toInt(&yok);
    if(xok) move(x, pos().y());
    if(xok && yok) move(x, y);
    arg.erase(i); arg.erase(i2);
  }

  // play/lastframe
  bool playArg=false, lastframeArg=false;
  if((i=std::find(arg.begin(), arg.end(), "--play"))!=arg.end())
    { playArg=true; arg.erase(i); }
  else if((i=std::find(arg.begin(), arg.end(), "--lastframe"))!=arg.end())
    { lastframeArg=true; arg.erase(i); }

  // speed
  if((i=std::find(arg.begin(), arg.end(), "--speed"))!=arg.end()) {
    i2=i; i2++;
    oldSpeed=QString(i2->c_str()).toDouble();
    speedSB->setValue(oldSpeed);
    arg.erase(i); arg.erase(i2);
  }

  // camera position
  string cameraFile;
  if((i=std::find(arg.begin(), arg.end(), "--camera"))!=arg.end()) {
    i2=i; i2++;
    cameraFile=*i2;
    arg.erase(i); arg.erase(i2);
  }

  // head light
  if((i=std::find(arg.begin(), arg.end(), "--headlight"))!=arg.end()) {
    i2=i; i2++;
    SoInput input;
    input.openFile(i2->c_str());
    SoBase *newHeadLight;
    if(SoBase::read(&input, newHeadLight, SoDirectionalLight::getClassTypeId())) {
      glViewer->getHeadlight()->on.setValue(((SoDirectionalLight*)newHeadLight)->on.getValue());
      glViewer->getHeadlight()->intensity.setValue(((SoDirectionalLight*)newHeadLight)->intensity.getValue());
      glViewer->getHeadlight()->color.setValue(((SoDirectionalLight*)newHeadLight)->color.getValue());
    }
    arg.erase(i); arg.erase(i2);
  }

  // auto reload
  reloadTimeout=0;
  if((i=std::find(arg.begin(), arg.end(), "--autoreload"))!=arg.end()) {
    i2=i; i2++;
    char *error;
    reloadTimeout=strtol(i2->c_str(), &error, 10);
    if(reloadTimeout<0) reloadTimeout=250;
    arg.erase(i);
    if(error && strlen(error)==0)
      arg.erase(i2);
    else
      reloadTimeout=250;
  }

  // maximized
  if((i=std::find(arg.begin(), arg.end(), "--maximized"))!=arg.end())
    showMaximized();

  // read XML files
  QDir dir;
  QRegExp filterRE1(".+\\.ombvx");
  dir.setFilter(QDir::Files);
  i=arg.begin();
  while(i!=arg.end()) {
    dir.setPath(i->c_str());
    if(dir.exists()) { // if directory
      // open all .+\.ombvx
      QStringList file=dir.entryList();
      for(int j=0; j<file.size(); j++)
        if(filterRE1.exactMatch(file[j]))
          openFile(dir.path().toStdString()+"/"+file[j].toStdString());
      i2=i; i++; arg.erase(i2);
      continue;
    }
    if(QFile::exists(i->c_str())) {
      if(openFile(*i)) {
        i2=i; i++; arg.erase(i2);
      }
      else
        i++;
      continue;
    }
    i++;
  }

  // arg commands after load all files
  
  // camera
  if(!cameraFile.empty()) {
    loadCamera(cameraFile);
  }
  else
    glViewer->viewAll();

  // play
  if(playArg) playAct->trigger();

  // lastframe
  if(lastframeArg) lastFrameAct->trigger();

  //accept drag and drop
  setAcceptDrops(true);

  // auto exit if everything is finished
  if(std::find(arg.begin(), arg.end(), "--autoExit")!=arg.end()) {
    auto timer=new QTimer(this);
    connect(timer, &QTimer::timeout, [this, timer](){
      if(waitFor.empty()) {
        timer->stop();
        if(!close())
          timer->start(100);
      }
    });
    timer->start(100);
  }
}

MainWindow* const MainWindow::getInstance() {
  return instance;
}

void MainWindow::disableBBox(Object *obj) {
  obj->setBoundingBox(false);
}
void MainWindow::highlightObject(Object *current) {
  // disable all bbox
  Utils::visitTreeWidgetItems<Object*>(objectList->invisibleRootItem(), &disableBBox);
  // enable current bbox
  current->setBoundingBox(true);
}
void MainWindow::enableBBoxOfID(Object *obj, const string &ID) {
  if(obj->object->getID()!=ID)
    return;
  obj->setBoundingBox(true);
}
void MainWindow::highlightObject(string curID) {
  // disable all bbox
  Utils::visitTreeWidgetItems<Object*>(objectList->invisibleRootItem(), &disableBBox);
  // enable all curID bbox
  if(!curID.empty())
    Utils::visitTreeWidgetItems<Object*>(objectList->invisibleRootItem(), std::bind(&enableBBoxOfID, _1, curID));
}

MainWindow::~MainWindow() {
  // unload all top level files before exit (from last to first since the unload removes the element from the list)
  for(int i=objectList->invisibleRootItem()->childCount()-1; i>=0; i--)
    ((Group*)(objectList->invisibleRootItem()->child(i)))->unloadFileSlot();
  cameraPosition->unref();
  sceneRoot->unref();
  timeString->unref();
  olseColor->unref();
  cameraOrientation->unref();
  olseDrawStyle->unref();
  delete fpsTime;
  delete time;
  delete glViewer;
  delete bgColor;
  delete fgColorTop;
  delete fgColorBottom;
  delete engDrawingBGColorSaved;
  delete engDrawingFGColorBottomSaved;
  delete engDrawingFGColorTopSaved;
  SoDB::renameGlobalField("frame", ""); // delete global field
  delete frameSensor;

  // delete all globally stored Coin data before deinit Coin/SoQt
  EdgeCalculation::edgeCache.clear();
  Utils::ivBodyCache.clear();
  SoQt::done();

  Utils::deinitialize();
}

bool MainWindow::openFile(const std::string& fileName, QTreeWidgetItem* parentItem, SoGroup *soParent, int ind) {
  // default parameter
  if(parentItem==nullptr) parentItem=objectList->invisibleRootItem();
  if(soParent==nullptr) soParent=sceneRoot;

  // read XML
  std::shared_ptr<OpenMBV::Group> rootGroup=OpenMBV::ObjectFactory::create<OpenMBV::Group>();
  rootGroup->setFileName(fileName);
  rootGroup->read();
  if(rootGroup->getHDF5File())
    rootGroup->getHDF5File()->refreshAfterWriterFlush();

  // Duplicate OpenMBVCppInterface tree using OpenMBV tree
  Object *object=ObjectFactory::create(rootGroup, parentItem, soParent, ind);
  object->setText(0, fileName.c_str());
  object->setToolTip(0, QFileInfo(fileName.c_str()).absoluteFilePath());
  object->getIconFile()="h5file.svg";
  object->setIcon(0, Utils::QIconCached(object->getIconFile()));

  // force a update
  frame->touch();
  // apply object filter
  objectListFilter->applyFilter();

  return true;
}

void MainWindow::openFileDialog() {
  QStringList files=QFileDialog::getOpenFileNames(nullptr, "Add OpenMBV Files", ".",
    "OpenMBV files (*.ombvx)");
  for(int i=0; i<files.size(); i++)
    openFile(files[i].toStdString());
}

void MainWindow::newFileDialog() {
  QFileDialog dialog;
  dialog.setWindowTitle("New OpenMBV File");
  dialog.setDirectory(".");
  dialog.setNameFilter(
    "OpenMBV files (*.ombvx)");
  dialog.setAcceptMode(QFileDialog::AcceptSave);
  dialog.setDefaultSuffix("ombx");
  if(dialog.exec()==QDialog::Rejected) return;

  boost::filesystem::path filename=dialog.selectedFiles()[0].toStdString();
  boost::filesystem::ofstream file(filename);
  file<<R"(<?xml version="1.0" encoding="UTF-8" ?>)"<<endl
      <<"<Group name=\""<<filename.stem().string()<<R"(" xmlns="http://www.mbsim-env.de/OpenMBV"/>)"<<endl;
  openFile(filename.string());
}

void MainWindow::toggleAction(Object *current, QAction *currentAct) {
  QList<QAction*> actions=current->getProperties()->getActions();
  for(auto & action : actions)
    if(action->objectName()==currentAct->objectName() && currentAct!=action)
      action->trigger();
}
void MainWindow::execPropertyMenu() {
  auto *object=(Object*)objectList->currentItem();
  QMenu* menu=object->getProperties()->getContextMenu();
  QAction *currentAct=menu->exec(QCursor::pos());
  // if action is not NULL and the action has a object name trigger also the actions with
  // the same name of all other selected objects
  if(currentAct && currentAct->objectName()!="")
    Utils::visitTreeWidgetItems<Object*>(objectList->invisibleRootItem(), std::bind(&toggleAction, _1, currentAct), true);
}

void MainWindow::objectListClicked() {
  if(QApplication::mouseButtons()==Qt::RightButton) {
    execPropertyMenu();
    frame->touch(); // force rendering the scene
  }
  if(!objectList->currentItem()) return;
  objectSelected(static_cast<Object*>(objectList->currentItem())->object->getID(), 
                 static_cast<Object*>(objectList->currentItem()));
}

void MainWindow::guiHelp() {
  static QDialog *gui=nullptr;
  if(gui==nullptr) {
    gui=new QDialog(this);
    gui->setWindowTitle("GUI Help");
    gui->setMinimumSize(500, 500);
    auto *layout=new QGridLayout;
    gui->setLayout(layout);
    auto *text=new QTextEdit;
    layout->addWidget(text, 0, 0);
    text->setReadOnly(true);
    boost::filesystem::ifstream file(boost::dll::program_location().parent_path().parent_path()/"share"/"openmbv"/"doc"/"guihelp.html");
    string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    text->setHtml(content.c_str());
  }
  gui->show();
}

void MainWindow::xmlHelp() {
  QDesktopServices::openUrl(QUrl::fromLocalFile((boost::dll::program_location().parent_path().parent_path()/"share"/"mbxmlutils"/"doc"/"http___www_mbsim-env_de_OpenMBV"/"index.html").string().c_str()));
}

void MainWindow::aboutOpenMBV() {
  static QDialog *about=nullptr;
  if(about==nullptr) {
    about=new QDialog(this);
    about->setWindowTitle("About OpenMBV");
    about->setMinimumSize(500, 500);
    auto *layout=new QGridLayout;
    layout->setColumnStretch(0, 0);
    layout->setColumnStretch(1, 1);
    about->setLayout(layout);
    QLabel *icon=new QLabel;
    layout->addWidget(icon, 0, 0, Qt::AlignTop);
    icon->setPixmap(Utils::QIconCached((boost::dll::program_location().parent_path().parent_path()/"share"/"openmbv"/"icons"/"openmbv.svg")
                    .string().c_str()).pixmap(64,64));
    auto *text=new QTextEdit;
    layout->addWidget(text, 0, 1);
    text->setReadOnly(true);
    text->setHtml(
      "<h1>OpenMBV - Open Multi Body Viewer</h1>"
      "<p>Copyright &copy; Markus Friedrich <tt>&lt;friedrich.at.gc@googlemail.com&gt;</tt><p/>"
      "<p>Licensed under the General Public License (see file COPYING).</p>"
      "<p>This is free software; see the source for copying conditions.  There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.</p>"
      "<h2>Authors:</h2>"
      "<ul>"
      "  <li>Markus Friedrich <tt>&lt;friedrich.at.gc@googlemail.com&gt;</tt> (Maintainer)</li>"
      "</ul>"
      "<h2>This program uses:</h2>"
      "<ul>"
      "  <li>'Qt - A cross-platform application and UI framework' by Nokia from <tt>http://www.qtsoftware.com</tt> (License: GPL/LGPL)</li>"
      "  <li>'Coin - An OpenGL based, retained mode 3D graphics rendering library' by Kongsberg SIM from <tt>http://www.coin3d.org</tt> (License: GPL)</li>"
      "  <li>'SoQt - A Qt GUI component toolkit library for Coin' by Kongsberg SIM from <tt>http://www.coin3d.org</tt> (License: GPL)</li>"
      "  <li>'HDF5Serie - A HDF5 Wrapper for Time Series' by Markus Friedrich from <tt>https://www.mbsim-env.de/</tt> (License: LGPL)</li>"
      "  <li>'HDF - Hierarchical Data Format' by The HDF Group from <tt>http://www.hdfgroup.org</tt> (License: NCSA-HDF)</li>"
      "  <li>'xerces-c - A validating XML parser' by Apache from <tt>http://xerces.apache.org/xerces-c</tt> (Licence: Apache)</li>"
      "  <li>'boost - C++ Libraries' by Boost from <tt>http://www.boost.org</tt> (Licence: Boost Software License)</li>"
      "  <li>'Qwt - Qt Widgets for Technical Applications' by Uwe Rathmann from <tt>http://qwt.sourceforge.net</tt> (Licence: Qwt/LGPL)</li>"
      "  <li>...</li>"
      "</ul>"
      "<p>A special thanks to all authors of this projects.</p>"
    );
  }
  about->show();
}

void MainWindow::viewChange(ViewSide side) {
  SbRotation r, r2;
  SbVec3f n;
  switch(side) {
    case top:
      glViewer->getCamera()->position.setValue(0,0,+1);
      glViewer->getCamera()->pointAt(SbVec3f(0,0,0), SbVec3f(0,1,0));
      glViewer->viewAll();
      break;
    case bottom:
      glViewer->getCamera()->position.setValue(0,0,-1);
      glViewer->getCamera()->pointAt(SbVec3f(0,0,0), SbVec3f(0,1,0));
      glViewer->viewAll();
      break;
    case front:
      glViewer->getCamera()->position.setValue(0,-1,0);
      glViewer->getCamera()->pointAt(SbVec3f(0,0,0), SbVec3f(0,0,1));
      glViewer->viewAll();
      break;
    case back:
      glViewer->getCamera()->position.setValue(0,+1,0);
      glViewer->getCamera()->pointAt(SbVec3f(0,0,0), SbVec3f(0,0,1));
      glViewer->viewAll();
      break;
    case right:
      glViewer->getCamera()->position.setValue(+1,0,0);
      glViewer->getCamera()->pointAt(SbVec3f(0,0,0), SbVec3f(0,0,1));
      glViewer->viewAll();
      break;
    case left:
      glViewer->getCamera()->position.setValue(-1,0,0);
      glViewer->getCamera()->pointAt(SbVec3f(0,0,0), SbVec3f(0,0,1));
      glViewer->viewAll();
      break;
    case isometric:
      glViewer->getCamera()->position.setValue(1,1,1);
      glViewer->getCamera()->pointAt(SbVec3f(0,0,0), SbVec3f(0,0,1));
      glViewer->viewAll();
      break;
    case dimetric:
      glViewer->getCamera()->orientation.setValue(Utils::cardan2Rotation(SbVec3f(-1.227769277146394,0,1.227393504015536)));
      glViewer->viewAll();
      break;
    case rotateXpWorld:
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(SbVec3f(-1,0,0), 10*M_PI/180);
      glViewer->getCamera()->orientation.setValue(r);
      break;
    case rotateXmWorld:
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(SbVec3f(-1,0,0), -10*M_PI/180);
      glViewer->getCamera()->orientation.setValue(r);
      break;
    case rotateYpWorld:
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(SbVec3f(0,-1,0), 10*M_PI/180);
      glViewer->getCamera()->orientation.setValue(r);
      break;
    case rotateYmWorld:
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(SbVec3f(0,-1,0), -10*M_PI/180);
      glViewer->getCamera()->orientation.setValue(r);
      break;
    case rotateZpWorld:
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(SbVec3f(0,0,-1), 10*M_PI/180);
      glViewer->getCamera()->orientation.setValue(r);
      break;
    case rotateZmWorld:
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(SbVec3f(0,0,-1), -10*M_PI/180);
      glViewer->getCamera()->orientation.setValue(r);
      break;
    case rotateXpScreen:
      r2=glViewer->getCamera()->orientation.getValue(); // camera orientation
      r2*=((SoSFRotation*)(cameraOrientation->outRotation[0]))->getValue(); // camera orientation relative to "Move Camera with Body"
      r2.multVec(SbVec3f(-1,0,0),n);
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(n, 10*M_PI/180);
      glViewer->getCamera()->orientation.setValue(r);
      break;
    case rotateXmScreen:
      r2=glViewer->getCamera()->orientation.getValue(); // camera orientation
      r2*=((SoSFRotation*)(cameraOrientation->outRotation[0]))->getValue(); // camera orientation relative to "Move Camera with Body"
      r2.multVec(SbVec3f(-1,0,0),n);
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(n, -10*M_PI/180);
      glViewer->getCamera()->orientation.setValue(r);
      break;
    case rotateYpScreen:
      r2=glViewer->getCamera()->orientation.getValue(); // camera orientation
      r2*=((SoSFRotation*)(cameraOrientation->outRotation[0]))->getValue(); // camera orientation relative to "Move Camera with Body"
      r2.multVec(SbVec3f(0,-1,0),n);
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(n, 10*M_PI/180);
      glViewer->getCamera()->orientation.setValue(r);
      break;
    case rotateYmScreen:
      r2=glViewer->getCamera()->orientation.getValue(); // camera orientation
      r2*=((SoSFRotation*)(cameraOrientation->outRotation[0]))->getValue(); // camera orientation relative to "Move Camera with Body"
      r2.multVec(SbVec3f(0,-1,0),n);
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(n, -10*M_PI/180);
      glViewer->getCamera()->orientation.setValue(r);
      break;
    case rotateZpScreen:
      r2=glViewer->getCamera()->orientation.getValue(); // camera orientation
      r2*=((SoSFRotation*)(cameraOrientation->outRotation[0]))->getValue(); // camera orientation relative to "Move Camera with Body"
      r2.multVec(SbVec3f(0,0,-1),n);
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(n, 10*M_PI/180);
      glViewer->getCamera()->orientation.setValue(r);
      break;
    case rotateZmScreen:
      r2=glViewer->getCamera()->orientation.getValue(); // camera orientation
      r2*=((SoSFRotation*)(cameraOrientation->outRotation[0]))->getValue(); // camera orientation relative to "Move Camera with Body"
      r2.multVec(SbVec3f(0,0,-1),n);
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(n, -10*M_PI/180);
      glViewer->getCamera()->orientation.setValue(r);
      break;
  }
}

static void clearSoEvent(SoEvent *ev) {
  ev->setShiftDown(false);
  ev->setCtrlDown(false);
  ev->setAltDown(false);
  if(ev->isOfType(SoButtonEvent::getClassTypeId())) {
    auto *bev=static_cast<SoButtonEvent*>(ev);
    bev->setState(SoButtonEvent::UNKNOWN);
  }
  if(ev->isOfType(SoMouseButtonEvent::getClassTypeId())) {
    auto *mbev=static_cast<SoMouseButtonEvent*>(ev);
    mbev->setButton(SoMouseButtonEvent::ANY);
  }
  if(ev->isOfType(SoKeyboardEvent::getClassTypeId())) {
    auto *kev=static_cast<SoKeyboardEvent*>(ev);
    kev->setKey(SoKeyboardEvent::UNDEFINED);
  }
}

bool MainWindow::soQtEventCB(const SoEvent *const event) {
  static Mode mode=no;

  if(event->isOfType(SoMouseButtonEvent::getClassTypeId())) {
    auto *ev=const_cast<SoMouseButtonEvent*>(static_cast<const SoMouseButtonEvent*>(event));
    static QPoint buttonDownPoint;
    static QElapsedTimer timer;
    // save point whre button down was pressed
    if(ev->getState()==SoButtonEvent::DOWN)
      buttonDownPoint=QCursor::pos();
    // detect a double click (at mouse up)
    bool doubleClick=false;
    if(ev->getState()==SoButtonEvent::UP) {
      if(timer.isValid() && timer.restart()<QApplication::doubleClickInterval())
        doubleClick=true;
      if(!timer.isValid())
        timer.start();
    }
    // button up without move of cursor => treat as button pressed
    // Do not return inside this code block: the button up event must be processed (to reselect mode, ...)
    if(ev->getState()==SoButtonEvent::UP && (QCursor::pos()-buttonDownPoint).manhattanLength()<=2) {
      // get picked points by ray
      SoRayPickAction pickAction(glViewer->getViewportRegion());
      pickAction.setPoint(ev->getPosition());
      pickAction.setRadius(3.0);
      pickAction.setPickAll(true);
      pickAction.apply(glViewer->getSceneManager()->getSceneGraph());
      SoPickedPointList pickedPoints=pickAction.getPickedPointList();
      // get objects by point/path
      list<Body*> pickedObject;
      float x=1e99, y=1e99, z=1e99;
      msg(Info)<<"Clicked points:\n";
      for(int i=0; pickedPoints[i]; i++) {
        SoPath *path=pickedPoints[i]->getPath();
        bool found=false;
        for(int j=path->getLength()-1; j>=0; j--) {
          auto it=Body::getBodyMap().find(path->getNode(j));
          if(it!=Body::getBodyMap().end()) {
            if(std::find(pickedObject.begin(), pickedObject.end(), it->second)==pickedObject.end())
              pickedObject.push_back(it->second);
            found=true;
            break;
          }
        }
        if(!found) continue;

        // get picked point and delete the cameraPosition and cameraOrientation values (if camera moves with body)
        SbVec3f delta;
        cameraOrientation->inRotation.getValue().multVec(pickedPoints[i]->getPoint(), delta);
        (delta+cameraPosition->vector[0]).getValue(x,y,z);

        QString str("Point [%1, %2, %3] on %4"); str=str.arg(x).arg(y).arg(z).arg((*(--pickedObject.end()))->object->getFullName(true, true).c_str());
        statusBar()->showMessage(str);
        msg(Info)<<str.toStdString()<<"\n";
      }
      msg(Info)<<endl;
      // mid button clicked => seed rotation center to clicked point
      if(ev->getButton()==SoMouseButtonEvent::BUTTON3 && pickedPoints[0]!=nullptr)
        glViewer->seekToPoint(pickedPoints[0]->getPoint());
      // if at least one object was picked
      if(!pickedObject.empty()) {
        bool objectClicked=false;
        // left or right button clicked => select object
        if(ev->getButton()==SoMouseButtonEvent::BUTTON1 || ev->getButton()==SoMouseButtonEvent::BUTTON2) {
          // Alt was down => show menu of all objects under the clicked point and select the clicked object of this menu
          if(ev->wasAltDown()) {
            auto *menu=new QMenu(this);
            int ind=0;
            list<Body*>::iterator it;
            for(it=pickedObject.begin(); it!=pickedObject.end(); it++) {
              QAction *action=new QAction((*it)->icon(0),(*it)->object->getFullName(true, true).c_str(),menu);
              action->setData(QVariant(ind++));
              menu->addAction(action);
            }
            QAction *action=menu->exec(QCursor::pos());
            if(action!=nullptr) {
              ind=action->data().toInt();
              it=pickedObject.begin();
              for(int i=0; i<ind; i++, it++);
              objectList->setCurrentItem(*it,0,ev->wasCtrlDown()?QItemSelectionModel::Toggle:QItemSelectionModel::ClearAndSelect);
              objectSelected((*it)->object->getID(), *it);
              objectClicked=true;
            }
            delete menu;
          }
          // alt was not down => select the first object under the clicked point
          else {
            objectList->setCurrentItem(*pickedObject.begin(),0,ev->wasCtrlDown()?QItemSelectionModel::Toggle:QItemSelectionModel::ClearAndSelect);
            objectSelected((*pickedObject.begin())->object->getID(), *pickedObject.begin());
            objectClicked=true;
          }
          // right button => show context menu of picked object
          if(ev->getButton()==SoMouseButtonEvent::BUTTON2 && objectClicked)
            execPropertyMenu();
        }
        // left button double clicked (the first click has alread select a object)
        if(ev->getButton()==SoMouseButtonEvent::BUTTON1 && doubleClick && objectClicked) {
          // return the current item if existing or the first selected item
          Object *object=static_cast<Object*>(objectList->currentItem()?objectList->currentItem():objectList->selectedItems().first());
          // show properties dialog only if objectDoubleClicked is not connected to some other slot
          if(!isSignalConnected(QMetaMethod::fromSignal(&MainWindow::objectDoubleClicked)))
            object->getProperties()->show();
          objectDoubleClicked(object->object->getID(), object);
        }
      }
    }
    // on scroll frame up/down
    if(ev->getButton()==SoMouseButtonEvent::BUTTON4) {
      frameSB->stepUp();
      return true;
    }
    if(ev->getButton()==SoMouseButtonEvent::BUTTON5) {
      frameSB->stepDown();
      return true;
    }
    // pass left button to SoQt as left button
    if(ev->getState()==SoButtonEvent::DOWN && ev->getButton()==SoMouseButtonEvent::BUTTON1) {
      clearSoEvent(ev);
      ev->setState(SoButtonEvent::DOWN);
      ev->setButton(SoMouseButtonEvent::BUTTON1);
      mode=rotate;
      return false;
    }
    if(ev->getState()==SoButtonEvent::UP && ev->getButton()==SoMouseButtonEvent::BUTTON1) {
      clearSoEvent(ev);
      ev->setState(SoButtonEvent::UP);
      ev->setButton(SoMouseButtonEvent::BUTTON1);
      mode=no;
      return false;
    }
    // pass right button to SoQt as mid button
    if(ev->getState()==SoButtonEvent::DOWN && ev->getButton()==SoMouseButtonEvent::BUTTON2) {
      clearSoEvent(ev);
      ev->setState(SoButtonEvent::DOWN);
      ev->setButton(SoMouseButtonEvent::BUTTON3);
      mode=translate;
      return false;
    }
    if(ev->getState()==SoButtonEvent::UP && ev->getButton()==SoMouseButtonEvent::BUTTON2) {
      clearSoEvent(ev);
      ev->setState(SoButtonEvent::UP);
      ev->setButton(SoMouseButtonEvent::BUTTON3);
      mode=no;
      return false;
    }
    // pass mid button to SoQt as ctrl+mid button
    if(ev->getState()==SoButtonEvent::DOWN && ev->getButton()==SoMouseButtonEvent::BUTTON3) {
      clearSoEvent(ev);
      ev->setCtrlDown(true);
      ev->setState(SoButtonEvent::DOWN);
      ev->setButton(SoMouseButtonEvent::BUTTON3);
      mode=zoom;
      return false;
    }
    if(ev->getState()==SoButtonEvent::UP && ev->getButton()==SoMouseButtonEvent::BUTTON3) {
      clearSoEvent(ev);
      ev->setCtrlDown(false);
      ev->setState(SoButtonEvent::UP);
      ev->setButton(SoMouseButtonEvent::BUTTON3);
      mode=no;
      return false;
    }
  }
  if(event->isOfType(SoLocation2Event::getClassTypeId())) {
    auto *ev=const_cast<SoLocation2Event*>(static_cast<const SoLocation2Event*>(event));
    // if mode==zoom is active pass to SoQt with ctrl down
    if(mode==zoom) {
      clearSoEvent(ev);
      ev->setCtrlDown(true);
      return false;
    }
    // other mode's pass to SoQt
    if(mode!=no) {
      clearSoEvent(ev);
      return false;
    }
  }
  return true;
}

void MainWindow::frameSensorCB(void *data, SoSensor*) {
  auto *me=(MainWindow*)data;
  me->setObjectInfo(me->objectList->currentItem());
  me->timeSlider->setValue(MainWindow::instance->getFrame()->getValue());
  me->frameSB->setValue(MainWindow::instance->getFrame()->getValue());
}

void MainWindow::fpsCB() {
  static int count=1;

  int dt=fpsTime->restart();
  if(dt==0) {
    count++;
    return;
  }

  float T=0.5; // PT1 time constant
  float fps_=1000.0/dt*count; // current fps
  static float fpsOut=0;
  fpsOut=(1/T+fpsOut)/(1+1.0/T/fps_); // PT1 filtered fps
  if(fpsMax>1e-15 && fpsOut>0.9*fpsMax)
    fps->setText(QString("FPS: >%1(max)").arg(fpsMax, 0, 'f', 1));
  else
    fps->setText(QString("FPS: %1").arg(fpsOut, 0, 'f', 1));

  count=1;
}

void MainWindow::restartPlay() {
  if(playAct->isChecked()) {
    // emulate anim stop click
    animTimer->stop();
    // emulate anim play click
    animStartFrame=frame->getValue();
    time->restart();
    if(fpsMax<1e-15)
      animTimer->start();
    else
      animTimer->start((int)(1000/fpsMax));
  }
}

void MainWindow::heavyWorkSlot() {
  if(playAct->isChecked()) {
    double dT=time->elapsed()/1000.0*speedSB->value();// time since play click
    auto dframe=(int)(dT/deltaTime);// frame increment since play click
    unsigned int frame_=(animStartFrame+dframe-timeSlider->currentMinimum()) %
                        (timeSlider->currentMaximum()-timeSlider->currentMinimum()+1) + timeSlider->currentMinimum(); // frame number
    if(frame->getValue()!=frame_) frame->setValue(frame_); // set frame => update scene
    //glViewer->render(); // force rendering
  }
  else if(lastFrameAct->isChecked()) {
    // get number of rows of first none enviroment body
    if(!openMBVBodyForLastFrame) {
      auto it=Body::getBodyMap().begin();
      while(it!=Body::getBodyMap().end() && std::static_pointer_cast<OpenMBV::Body>(it->second->object)->getRows()==0)
        it++;
      openMBVBodyForLastFrame=std::static_pointer_cast<OpenMBV::Body>(it->second->object);
    }
    // refresh all files
    H5::File::refreshAllFilesAfterWriterFlush();
    // use number of rows for found first none enviroment body
    int currentNumOfRows=openMBVBodyForLastFrame->getRows();
    if(deltaTime==0 && currentNumOfRows>=2)
      deltaTime=openMBVBodyForLastFrame->getRow(1)[0]-openMBVBodyForLastFrame->getRow(0)[0];
    if(currentNumOfRows<2) return;

    // update if a new row is available
    if(currentNumOfRows-2!=timeSlider->totalMaximum()) {
      timeSlider->setTotalMaximum(currentNumOfRows-2);
      timeSlider->setCurrentMaximum(currentNumOfRows-2);
      frame->setValue(currentNumOfRows-2);
    }
  }
}

void MainWindow::speedWheelChanged(int value) {
  speedSB->setValue(oldSpeed*pow(10,value/10000.0));
}

void MainWindow::speedWheelPressed() {
  oldSpeed=speedSB->value();
}

void MainWindow::speedWheelReleased() {
  oldSpeed=speedSB->value();
  speedWheel->setValue(0);
}

void MainWindow::exportAsPNG(short width, short height, const std::string& fileName, bool transparent) {
  SoOffscreenRenderer myRenderer(SbViewportRegion(width, height));
  myRenderer.setViewportRegion(SbViewportRegion(width, height));
  if(transparent)
    myRenderer.setComponents(SoOffscreenRenderer::RGB_TRANSPARENCY);
  else
    myRenderer.setComponents(SoOffscreenRenderer::RGB);

  // root separator for export
  auto *root=new SoSeparator;
  root->ref();
  SbColor fgColorTopSaved=*fgColorTop->getValues(0);
  SbColor fgColorBottomSaved=*fgColorBottom->getValues(0);
  // add background
  if(!transparent) {
    // do not write to depth buffer
    auto *db1=new SoDepthBuffer;
    root->addChild(db1);
    db1->write.setValue(false);
    // render background
    root->addChild(glViewer->bgSep);
    // write to depth buffer until now
    auto *db2=new SoDepthBuffer;
    root->addChild(db2);
    db2->write.setValue(true);
  }
  // set text color to black
  else {
    fgColorTop->set1Value(0, 0,0,0);
    fgColorBottom->set1Value(0, 0,0,0);
  }
  // add scene
  root->addChild(glViewer->getSceneManager()->getSceneGraph());
  // do not test depth buffer
  auto *db=new SoDepthBuffer;
  root->addChild(db);
  db->function.setValue(SoDepthBufferElement::ALWAYS);
  // add foreground
  root->addChild(glViewer->fgSep);
  // update/redraw glViewer: this is required to update e.g. the clipping planes before offscreen rendering
  // (SoOffscreenRenderer does not update the clipping planes but SoQtViewer does so!)
  // (it gives the side effect, that the user sees the current exported frame)
  // (the double rendering does not lead to permormance problems)
  glViewer->redraw();
  // render offscreen
  SbBool ok=myRenderer.render(root);
  if(!ok) {
    QMessageBox::warning(this, "PNG export warning", "Unable to render offscreen image. See OpenGL/Coin messages in console!");
    root->unref();
    return;
  }

  // set set text color
  if(transparent) {
    fgColorTop->set1Value(0, fgColorTopSaved);
    fgColorBottom->set1Value(0, fgColorBottomSaved);
  }

  auto *buf=new unsigned char[width*height*4];
  for(int y=0; y<height; y++)
    for(int x=0; x<width; x++) {
      int i=(y*width+x)* (transparent?4:3);
      int o=((height-y-1)*width+x)*4;
      buf[o+0]=myRenderer.getBuffer()[i+2]; // blue
      buf[o+1]=myRenderer.getBuffer()[i+1]; // green
      buf[o+2]=myRenderer.getBuffer()[i+0]; // red
      buf[o+3]=(transparent?myRenderer.getBuffer()[i+3]:255); // alpha
    }
  QImage image(buf, width, height, QImage::Format_ARGB32);
  image.save(fileName.c_str(), "png");
  delete[]buf;
  root->unref();
}

void MainWindow::exportCurrentAsPNG() {
  ExportDialog dialog(this, false);
  dialog.exec();
  if(dialog.result()==QDialog::Rejected) return;

  QString str("Exporting current frame, please wait!");
  statusBar()->showMessage(str);
  msg(Info)<<str.toStdString()<<endl;
  SbVec2s size=glViewer->getSceneManager()->getViewportRegion().getWindowSize()*dialog.getScale();
  short width, height; size.getValue(width, height);
  glViewer->font->size.setValue(glViewer->font->size.getValue()*dialog.getScale());
  exportAsPNG(width, height, dialog.getFileName().toStdString(), dialog.getTransparent());
  glViewer->font->size.setValue(glViewer->font->size.getValue()/dialog.getScale());
  str="Done";
  statusBar()->showMessage(str, 10000);
  msg(Info)<<str.toStdString()<<endl;
}

void MainWindow::exportSequenceAsPNG() {
  ExportDialog dialog(this, true);
  dialog.exec();
  if(dialog.result()==QDialog::Rejected) return;
  double scale=dialog.getScale();
  bool transparent=dialog.getTransparent();
  QString fileName=dialog.getFileName();
  if(!fileName.toUpper().endsWith(".PNG")) return;
  fileName=fileName.remove(fileName.length()-6,6);
  double speed=speedSB->value();
  int startFrame=timeSlider->currentMinimum();
  int endFrame=timeSlider->currentMaximum();
  double fps=dialog.getFPS();

  if(speed/deltaTime/fps<1) {
    int ret=QMessageBox::warning(this, "Export PNG sequence",
      "Some video-frames would contain the same data,\n"
      "because the animation speed is to slow,\n"
      "and/or the framerate is to high\n"
      "and/or the time-intervall of the data files is to large.\n"
      "\n"
      "Continue anyway?", QMessageBox::Yes, QMessageBox::No);
    if(ret==QMessageBox::No) return;
  }
  int videoFrame=0;
  auto lastVideoFrame=(int)(deltaTime*fps/speed*(endFrame-startFrame));
  SbVec2s size=glViewer->getSceneManager()->getViewportRegion().getWindowSize()*scale;
  short width, height; size.getValue(width, height);
  glViewer->font->size.setValue(glViewer->font->size.getValue()*dialog.getScale());
  for(int frame_=startFrame; frame_<=endFrame; frame_=(int)(speed/deltaTime/fps*++videoFrame+startFrame)) {
    QString str("Exporting frame sequence, please wait! (%1\%)"); str=str.arg(100.0*videoFrame/lastVideoFrame,0,'f',1);
    statusBar()->showMessage(str);
    msg(Info)<<str.toStdString()<<endl;
    frame->setValue(frame_);
    exportAsPNG(width, height, QString("%1_%2.png").arg(fileName).arg(videoFrame, 6, 10, QChar('0')).toStdString(), transparent);
  }
  glViewer->font->size.setValue(glViewer->font->size.getValue()/dialog.getScale());
  QString str("Done");
  statusBar()->showMessage(str, 10000);
  msg(Info)<<str.toStdString()<<endl;
}

void MainWindow::stopSCSlot() {
  animTimer->stop();
  stopAct->setChecked(true);
  lastFrameAct->setChecked(false);
  playAct->setChecked(false);
}

void MainWindow::lastFrameSCSlot() {
  if(!lastFrameAct->isChecked()) {
    stopAct->setChecked(true);
    animTimer->stop();
    return;
  }

  openMBVBodyForLastFrame=std::shared_ptr<OpenMBV::Body>();

  stopAct->setChecked(false);
  playAct->setChecked(false);
  if(fpsMax<1e-15)
    animTimer->start();
  else
    animTimer->start((int)(1000/fpsMax));
}

void MainWindow::playSCSlot() {
  if(!playAct->isChecked()) {
    stopAct->setChecked(true);
    animTimer->stop();
    return;
  }

  stopAct->setChecked(false);
  lastFrameAct->setChecked(false);
  animStartFrame=frame->getValue();
  time->restart();
  if(fpsMax<1e-15)
    animTimer->start();
  else
    animTimer->start((int)(1000/fpsMax));
}

void MainWindow::speedUpSlot() {
  speedSB->setValue(speedSB->value()*1.1);
}

void MainWindow::speedDownSlot() {
  speedSB->setValue(speedSB->value()/1.1);
}

void MainWindow::topBGColor() {
  float r,g,b;
  (*bgColor)[2].getValue(r,g,b);
  QColor color=QColorDialog::getColor(QColor((int)(r*255),(int)(g*255),(int)(b*255)));
  if(!color.isValid()) return;
  QRgb rgb=color.rgb();
  bgColor->set1Value(2, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
  bgColor->set1Value(3, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
  fgColorTop->set1Value(0, 1-(color.value()+127)/255,1-(color.value()+127)/255,1-(color.value()+127)/255);
}

void MainWindow::bottomBGColor() {
  float r,g,b;
  (*bgColor)[0].getValue(r,g,b);
  QColor color=QColorDialog::getColor(QColor((int)(r*255),(int)(g*255),(int)(b*255)));
  if(!color.isValid()) return;
  QRgb rgb=color.rgb();
  bgColor->set1Value(0, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
  bgColor->set1Value(1, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
  fgColorBottom->set1Value(0, 1-(color.value()+127)/255,1-(color.value()+127)/255,1-(color.value()+127)/255);
}

void MainWindow::olseColorSlot() {
  float r,g,b;
  olseColor->rgb.getValues(0)->getValue(r,g,b);
  QColor color=QColorDialog::getColor(QColor((int)(r*255),(int)(g*255),(int)(b*255)));
  if(!color.isValid()) return;
  QRgb rgb=color.rgb();
  olseColor->rgb.set1Value(0, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
}

void MainWindow::olseLineWidthSlot() {
  olseDrawStyle->lineWidth.setValue(QInputDialog::getDouble(this, "Outline and shilouette edge...", "Line width: ", olseDrawStyle->lineWidth.getValue()));
}

void MainWindow::loadWindowState() {
  loadWindowState("");
}
void MainWindow::loadWindowState(string filename) {
  if(filename.empty()) {
    QString fn=QFileDialog::getOpenFileName(nullptr, "Load window state", ".", "*.ombvwst");
    if(fn.isNull()) return;
    filename=fn.toStdString();
  }
  // load
  QFile stateFile(filename.c_str());
  stateFile.open(QIODevice::ReadOnly);
  WindowState windowState;
  stateFile.read((char*)&windowState, sizeof(WindowState));
  QByteArray wst=stateFile.readAll();
  stateFile.close();
  // geometry
  if(!windowState.hasMenuBar) toggleMenuBar->trigger();
  if(!windowState.hasStatusBar) toggleStatusBar->trigger();
  if(!windowState.hasFrameSlider) toggleFrameSlider->trigger();
  // window state
  restoreState(wst);
}

void MainWindow::saveWindowState() {
  QString str("--geometry %1x%2+%3+%4");
  str=str.arg(size().width()).arg(size().height()).
          arg(pos().x()).arg(pos().y());
  statusBar()->showMessage(str, 10000);
  msg(Info)<<str.toStdString()<<endl;

  QString filename=QFileDialog::getSaveFileName(nullptr, "Save window state", "openmbv.ombvwst", "*.ombvwst");
  if(filename.isNull()) return;
  if(!filename.endsWith(".ombvwst",Qt::CaseInsensitive))
    filename=filename+".ombvwst";
  // geometry
  WindowState windowState;
  windowState.hasMenuBar=toggleMenuBar->isChecked();
  windowState.hasStatusBar=toggleStatusBar->isChecked();
  windowState.hasFrameSlider=toggleFrameSlider->isChecked();
  QByteArray data((char*)&windowState, sizeof(WindowState));
  // window state
  QByteArray wst=saveState();
  data.append(wst);
  // save
  QFile stateFile(filename);
  stateFile.open(QIODevice::WriteOnly);
  stateFile.write(data);
  stateFile.close();
}

void MainWindow::loadCamera() {
  loadCamera("");
}
void MainWindow::loadCamera(string filename) {
  if(filename.empty()) {
    QString fn=QFileDialog::getOpenFileName(nullptr, "Load camera", ".", "*.camera.iv");
    if(fn.isNull()) return;
    filename=fn.toStdString();
  }
  SoInput input;
  input.openFile(filename.c_str());
  SoBase *newCamera;
  SoBase::read(&input, newCamera, SoCamera::getClassTypeId());
  if(newCamera->getTypeId()==SoOrthographicCamera::getClassTypeId()) {
    glViewer->setCameraType(SoOrthographicCamera::getClassTypeId());
    glViewer->myChangeCameraValues((SoCamera*)newCamera);
  }
  else if(newCamera->getTypeId()==SoPerspectiveCamera::getClassTypeId()) {
    glViewer->setCameraType(SoPerspectiveCamera::getClassTypeId());
    glViewer->myChangeCameraValues((SoCamera*)newCamera);
  }
  else {
    QString str("Only SoPerspectiveCamera and SoOrthographicCamera are allowed!");
    statusBar()->showMessage(str, 10000);
    msg(Info)<<str.toStdString()<<endl;
  }
}

void MainWindow::saveCamera() {
  QString filename=QFileDialog::getSaveFileName(nullptr, "Save camera", "openmbv.camera.iv", "*.camera.iv");
  if(filename.isNull()) return;
  if(!filename.endsWith(".camera.iv",Qt::CaseInsensitive))
    filename=filename+".camera.iv";
  SoOutput output;
  output.openFile(filename.toStdString().c_str());
  SoWriteAction wa(&output);
  wa.apply(glViewer->getCamera());
}

void MainWindow::exportCurrentAsIV() {
  QString filename=QFileDialog::getSaveFileName(nullptr, "Save current frame as iv", "openmbv.iv", "*.iv");
  if(filename.isNull()) return;
  if(!filename.endsWith(".iv",Qt::CaseInsensitive))
    filename=filename+".iv";
  SoOutput output;
  output.openFile(filename.toStdString().c_str());
  SoWriteAction wa(&output);
  wa.apply(sceneRoot);
}

void MainWindow::exportCurrentAsPS() {
  QString filename=QFileDialog::getSaveFileName(nullptr, "Save current frame as PS", "openmbv.ps", "*.ps");
  if(filename.isNull()) return;
  if(!filename.endsWith(".ps",Qt::CaseInsensitive))
    filename=filename+".ps";
  // set text color to black
  SbColor fgColorTopSaved=*fgColorTop->getValues(0);
  SbColor fgColorBottomSaved=*fgColorBottom->getValues(0);
  fgColorTop->set1Value(0, 0,0,0);
  fgColorBottom->set1Value(0, 0,0,0);
  // export in vector format
  SoVectorizePSAction ps;
  SoVectorOutput *out=ps.getOutput();
  out->openFile(filename.toStdString().c_str());
  short width, height;
  glViewer->getSceneManager()->getViewportRegion().getWindowSize().getValue(width, height);
  ps.beginPage(SbVec2f(0, 0), SbVec2f(width, height));
  ps.calibrate(glViewer->getViewportRegion());
  // root separator for export
  auto *root=new SoSeparator;
  root->ref();
  root->addChild(glViewer->getSceneManager()->getSceneGraph());
  root->addChild(glViewer->fgSep);
  ps.apply(root);
  ps.endPage();
  out->closeFile();
  root->unref();
  // reset text color
  fgColorTop->set1Value(0, fgColorTopSaved);
  fgColorBottom->set1Value(0, fgColorBottomSaved);
}

void MainWindow::toggleMenuBarSlot() {
  if(toggleMenuBar->isChecked())
    menuBar()->setVisible(true);
  else
    menuBar()->setVisible(false);
}

void MainWindow::toggleStatusBarSlot() {
  if(toggleStatusBar->isChecked())
    statusBar()->setVisible(true);
  else
    statusBar()->setVisible(false);
}

void MainWindow::toggleFrameSliderSlot() {
  if(toggleFrameSlider->isChecked())
    timeSlider->setVisible(true);
  else
    timeSlider->setVisible(false);
}

void MainWindow::toggleFullScreenSlot() {
  if(isFullScreen())
    showNormal();
  else
    showFullScreen();
}

void MainWindow::toggleDecorationSlot() {
  if(toggleDecoration->isChecked())
    setWindowFlags(Qt::Window);
  else
    setWindowFlags(Qt::FramelessWindowHint);
  show();
}

void MainWindow::releaseCameraFromBodySlot() {
  cameraPosition->vector.disconnect();
  cameraPosition->vector.setValue(0,0,0);
  cameraOrientation->inRotation.disconnect();
  cameraOrientation->inRotation.setValue(0,0,0,1);
  frame->touch(); // enforce update
}

void MainWindow::moveCameraWith(SoSFVec3f *pos, SoSFRotation *rot) {
  cameraPosition->vector.connectFrom(pos);
  cameraOrientation->inRotation.connectFrom(rot);
  frame->touch(); // enforce update
}

void MainWindow::showWorldFrameSlot() {
  if(worldFrameSwitch->whichChild.getValue()==SO_SWITCH_NONE)
    worldFrameSwitch->whichChild.setValue(SO_SWITCH_ALL);
  else
    worldFrameSwitch->whichChild.setValue(SO_SWITCH_NONE);
}

void MainWindow::setOutLineAndShilouetteEdgeRecursive(QTreeWidgetItem *obj, bool enableOutLine, bool enableShilouetteEdge) {
  for(int i=0; i<obj->childCount(); i++) {
    auto *act=((Object*)obj->child(i))->findChild<QAction*>("Body::outLine");
    if(act) act->setChecked(enableOutLine);
    act=((Object*)obj->child(i))->findChild<QAction*>("Body::shilouetteEdge");
    if(act) act->setChecked(enableShilouetteEdge);
    setOutLineAndShilouetteEdgeRecursive(obj->child(i), enableOutLine, enableShilouetteEdge);
  }
}
void MainWindow::toggleEngDrawingViewSlot() {
  if(engDrawingView->isChecked()) {
    // save bg color
    *engDrawingBGColorSaved=*bgColor;
    *engDrawingFGColorBottomSaved=*fgColorBottom;
    *engDrawingFGColorTopSaved=*fgColorTop;
    // set new bg color
    bgColor->set1Value(0, 1.0,1.0,1.0);
    bgColor->set1Value(1, 1.0,1.0,1.0);
    bgColor->set1Value(2, 1.0,1.0,1.0);
    bgColor->set1Value(3, 1.0,1.0,1.0);
    fgColorBottom->set1Value(0, 0,0,0);
    fgColorTop->set1Value(0, 0,0,0);
    engDrawing->whichChild.setValue(SO_SWITCH_ALL); // enable engineering drawing
    setOutLineAndShilouetteEdgeRecursive(objectList->invisibleRootItem(), true, true); // enable outline and shilouetteEdge
    glViewer->getCamera()->orientation.touch(); // redraw, like for a camera change
    topBGColorAct->setEnabled(false);
    bottomBGColorAct->setEnabled(false);
  }
  else {
    *bgColor=*engDrawingBGColorSaved; // restore bg color
    *fgColorBottom=*engDrawingFGColorBottomSaved;
    *fgColorTop=*engDrawingFGColorTopSaved;
    engDrawing->whichChild.setValue(SO_SWITCH_NONE); // disable engineering drawing
    setOutLineAndShilouetteEdgeRecursive(objectList->invisibleRootItem(), true, false); // enable outline and disable shilouetteEdge
    topBGColorAct->setEnabled(true);
    bottomBGColorAct->setEnabled(true);
  }
}

void MainWindow::complexityType() {
  QStringList typeItems;
  typeItems<<"Object space"<<"Screen space"<<"Bounding box";
  int current=0;
  if(complexity->type.getValue()==SoComplexity::OBJECT_SPACE) current=0;
  if(complexity->type.getValue()==SoComplexity::SCREEN_SPACE) current=1;
  if(complexity->type.getValue()==SoComplexity::BOUNDING_BOX) current=2;
  QString typeStr=QInputDialog::getItem(this, "Complexity...", "Type: ", typeItems, current, false);
  if(typeStr=="Object space") complexity->type.setValue(SoComplexity::OBJECT_SPACE);
  if(typeStr=="Screen space") complexity->type.setValue(SoComplexity::SCREEN_SPACE);
  if(typeStr=="Bounding box") complexity->type.setValue(SoComplexity::BOUNDING_BOX);
}

void MainWindow::complexityValue() {
  complexity->value.setValue(QInputDialog::getDouble(this, "Complexity...", "Value: ", complexity->value.getValue()*100, 0, 100, 1)/100);
}

void MainWindow::collapseItem(QTreeWidgetItem* item) {
  // a collapse/expandable item may be a Group or a CompoundRigidBody
  auto *grp=dynamic_cast<Group*>(item);
  if(grp)
    grp->grp->setExpand(false);
  auto *crb=dynamic_cast<CompoundRigidBody*>(item);
  if(crb)
    crb->crb->setExpand(false);
}

void MainWindow::expandItem(QTreeWidgetItem* item) {
  // a collapse/expandable item may be a Group or a CompoundRigidBody
  auto *grp=dynamic_cast<Group*>(item);
  if(grp)
    grp->grp->setExpand(true);
  auto *crb=dynamic_cast<CompoundRigidBody*>(item);
  if(crb)
    crb->crb->setExpand(true);
}

void MainWindow::editFinishedSlot() {
  QTreeWidgetItem *item=objectList->currentItem();
  static_cast<Object*>(item)->object->setName(item->text(0).toStdString());
}

void MainWindow::frameMinMaxSetValue(int min, int max) {
  frameMinSB->setRange(timeSlider->totalMinimum(), timeSlider->totalMaximum());
  frameMaxSB->setRange(timeSlider->totalMinimum(), timeSlider->totalMaximum());
  frameMinSB->setValue(min);
  frameMaxSB->setValue(max);
}

void MainWindow::selectionChanged() {
  QList<QTreeWidgetItem*> list=objectList->selectedItems();
  for(auto & i : list)
    static_cast<Object*>(i)->object->setSelected(i->isSelected());
}


void MainWindow::dragEnterEvent(QDragEnterEvent *event) {
  if (event->mimeData()->hasUrls()) {
    event->acceptProposedAction();
  }
}

void MainWindow::dropEvent(QDropEvent *event) {
  for (int i = 0; i < event->mimeData()->urls().size(); i++) {
    QString path = event->mimeData()->urls()[i].toLocalFile().toLocal8Bit().data();
    if (path.endsWith(".ombvx")) {
      QFile Fout(path);
      if (Fout.exists()) {
        openFile(Fout.fileName().toStdString());
      }
    }
  }
}

void MainWindow::closeEvent(QCloseEvent *event) {
  QSettings settings;
  settings.setValue("mainwindow/geometry", saveGeometry());
  settings.setValue("mainwindow/state", saveState());
  QMainWindow::closeEvent(event);
}

void MainWindow::showEvent(QShowEvent *event) {
  QSettings settings;
  restoreGeometry(settings.value("mainwindow/geometry").toByteArray());
  restoreState(settings.value("mainwindow/state").toByteArray());
}

}
