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

#include "config.h"
#include <openmbvcppinterface/group.h>
#include <openmbvcppinterface/cube.h>
#include <openmbvcppinterface/compoundrigidbody.h>
#include "mainwindow.h"
#include "mytouchwidget.h"
#include <algorithm>
#include <Inventor/Qt/SoQt.h>
#include <QDesktopWidget>
#include <QDesktopServices>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QInputDialog>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QApplication>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QColorDialog>
#include <QtCore/QElapsedTimer>
#include <QtCore/QTemporaryDir>
#include <QShortcut>
#include <QMimeData>
#include <QScroller>
#include "utils.h"
#include <QMetaMethod>
#include <QTapAndHoldGesture>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoCone.h>
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <Inventor/nodes/SoRotation.h>
#include <Inventor/nodes/SoDirectionalLight.h>
#include <Inventor/VRMLnodes/SoVRMLDirectionalLight.h>
#include <Inventor/nodes/SoLightModel.h>
#include <Inventor/nodes/SoSphere.h>
#include <Inventor/nodes/SoAnnotation.h>
#include <Inventor/nodes/SoDepthBuffer.h>
#include <Inventor/nodes/SoPolygonOffset.h>
#include <Inventor/nodes/SoShapeHints.h>
#include <Inventor/nodes/SoPointSet.h>
#include <Inventor/sensors/SoFieldSensor.h>
#include <Inventor/actions/SoWriteAction.h>
#include <Inventor/nodes/SoComplexity.h>
#include <Inventor/annex/HardCopy/SoVectorizePSAction.h>
#include <Inventor/engines/SoGate.h>
#include <Inventor/engines/SoCalculator.h>
#include <Inventor/fields/SoMFRotation.h>
#include "exportdialog.h"
#include "object.h"
#include "cuboid.h"
#include "group.h"
#include "objectfactory.h"
#include "compoundrigidbody.h"
#include <memory>
#include <string>
#include <set>
#include <hdf5serie/file.h>
#include <Inventor/SbViewportRegion.h>
#include <Inventor/actions/SoRayPickAction.h>
#include <Inventor/SoPickedPoint.h>
#include "IndexedTesselationFace.h"
#include "utils.h"
#include "touchtreewidget.h"
#include <boost/dll.hpp>

using namespace std;

namespace OpenMBVGUI {

MainWindow *MainWindow::instance=nullptr;

QObject* qTreeWidgetItemToQObject(const QModelIndex &index) {
  return dynamic_cast<QObject*>(static_cast<QTreeWidgetItem*>(index.internalPointer()));
}

MainWindow::MainWindow(list<string>& arg, bool _skipWindowState) : fpsMax(25), enableFullScreen(false),
  skipWindowState(_skipWindowState), deltaTime(0), oldSpeed(1) {
  OpenMBVGUI::appSettings=std::make_unique<OpenMBVGUI::AppSettings>();
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

  shortAniTimer=new QTimer(this);
  shortAniTimer->setInterval(1000/25);
  shortAniElapsed=std::make_unique<QElapsedTimer>();
  connect(shortAniTimer, &QTimer::timeout, this, &MainWindow::shortAni );

  // main widget
  auto *mainWG=new QWidget(this);
  setCentralWidget(mainWG);
  mainLO=new QGridLayout(mainWG);
  mainLO->setContentsMargins(0,0,0,0);
  mainWG->setLayout(mainLO);
  // gl viewer
  glViewerWG=new MyTouchWidget(this);
  timeString=new SoAsciiText;
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
  transparency=1;
  if((i=std::find(arg.begin(), arg.end(), "--transparency"))!=arg.end()) {
    i2=i; i2++;
    transparency=QString(i2->c_str()).toDouble();
    arg.erase(i); arg.erase(i2);
  }
  glViewer=new SoQtMyViewer(glViewerWG, transparency);
  sceneRoot=new SoSeparator;
  sceneRoot->ref();

  // 3D cursor
  auto *cursorAno=new SoAnnotation;
  sceneRoot->addChild(cursorAno);
  auto *cursorSep=new SoSepNoPickNoBBox;
  cursorAno->addChild(cursorSep);
  cursorSwitch=new SoSwitch;
  cursorSep->addChild(cursorSwitch);
  cursorPos=new SoTranslation;
  cursorSwitch->addChild(cursorPos);
  auto *cursorDrawStyle=new SoDrawStyle;
  cursorSwitch->addChild(cursorDrawStyle);
  cursorDrawStyle->lineWidth.setValue(3);
  SoScale *cursorScale, *cursorScale2;
  mouseCursorSizeField.setValue(appSettings->get<double>(AppSettings::mouseCursorSize));
  cursorScaleE=new SoCalculator;
  cursorSwitch->addChild(Utils::soFrame(0.5, 0.5, false, cursorScale, SbColor(1,1,1), SbColor(1,1,1), SbColor(1,1,1)));
  cursorScale->scaleFactor.connectFrom(&cursorScaleE->oA);
  auto *cursorDrawStyle2=new SoDrawStyle;
  cursorSwitch->addChild(cursorDrawStyle2);
  cursorDrawStyle2->lineWidth.setValue(1);
  cursorDrawStyle2->pointSize.setValue(4);
  cursorSwitch->addChild(Utils::soFrame(0.5, 0.5, false, cursorScale2));
  cursorScale2->scaleFactor.connectFrom(&cursorScaleE->oA);
  auto *cursorPointCoord=new SoCoordinate3;
  cursorSwitch->addChild(cursorPointCoord);
  cursorPointCoord->point.setValue(0,0,0);
  auto cursorPointCol=new SoBaseColor;
  cursorPointCol->rgb=SbColor(0,0,0);
  cursorSwitch->addChild(cursorPointCol);
  auto *cursorPoint=new SoPointSet;
  cursorSwitch->addChild(cursorPoint);

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
  drawStyle->linePattern.setValue(0b1111111111100000);
  worldFrameSep->addChild(Utils::soFrame(1,1,false));
  glViewer->setSceneGraph(sceneRoot);
  
  // time slider
  timeSlider=new QTripleSlider(this);
  mainLO->addWidget(timeSlider, 0, 1);
  timeSlider->setTotalRange(0, 0);
  connect(timeSlider, &QTripleSlider::sliderMoved, this, &MainWindow::updateFrame);

  // object list dock widget
  auto *objectListDW=new QDockWidget(tr("Objects"),this);
  objectListDW->setObjectName("MainWindow::objectListDW");
  auto *objectListWG=new QWidget(this);
  auto *objectListLO=new QGridLayout(objectListWG);
  objectListWG->setLayout(objectListLO);
  objectListDW->setWidget(objectListWG);
  addDockWidget(Qt::LeftDockWidgetArea,objectListDW);
  objectList = new TouchTreeWidget(objectListDW);
  objectListFilter=new AbstractViewFilter(objectList, 0, -1, "OpenMBVGUI::", &qTreeWidgetItemToQObject);
  objectListLO->addWidget(objectListFilter, 0,0);
  objectListLO->addWidget(objectList, 1,0);
  objectList->setHeaderHidden(true);
  objectList->setSelectionMode(QAbstractItemView::ExtendedSelection);
  objectList->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
  Utils::enableTouch(objectList);
  objectList->setContextMenuPolicy(Qt::CustomContextMenu);
  connect(objectList,&QTreeWidget::customContextMenuRequested,this, [this](const QPoint &pos){
    execPropertyMenu();
    frame->touch(); // force rendering the scene
  });
  connect(objectList,&QTreeWidget::pressed, this, &MainWindow::objectListClicked);
  connect(objectList,&QTreeWidget::itemDoubleClicked,this, [this](QTreeWidgetItem *item){
    if(!isSignalConnected(QMetaMethod::fromSignal(&MainWindow::objectDoubleClicked)))
      static_cast<Object*>(item)->getProperties()->openDialogSlot();
  });
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
  auto *objectInfoDW=new QDockWidget(tr("Object Info"),this);
  objectInfoDW->setObjectName("MainWindow::objectInfoDW");
  auto *objectInfoWG=new QWidget;
  auto *objectInfoLO=new QGridLayout;
  objectInfoWG->setLayout(objectInfoLO);
  objectInfoDW->setWidget(objectInfoWG);
  addDockWidget(Qt::LeftDockWidgetArea,objectInfoDW);
  objectInfo = new QTextEdit(objectInfoDW);
  objectInfoLO->addWidget(objectInfo, 0,0);
  objectInfo->setReadOnly(true);
  objectInfo->setLineWrapMode(QTextEdit::NoWrap);
  Utils::enableTouch(objectInfo);
  connect(objectList,&QTreeWidget::currentItemChanged,this,&MainWindow::setObjectInfo);

  // menu bar
  auto *mb=new QMenuBar(this);
  setMenuBar(mb);

  QAction *act;
  // file menu
  auto *fileMenu=new QMenu("File", menuBar());
  QAction *addFileAct=fileMenu->addAction(Utils::QIconCached("addfile.svg"), "Add file...", this, &MainWindow::openFileDialog);
  fileMenu->addAction(Utils::QIconCached("newfile.svg"), "New file...", this, &MainWindow::newFileDialog);
  fileMenu->addSeparator();
  act=fileMenu->addAction(Utils::QIconCached("exportimg.svg"), "Export current frame as PNG...", this, &MainWindow::exportCurrentAsPNG);
  addAction(act); // must work also if menu bar is invisible
  act=fileMenu->addAction(Utils::QIconCached("exportimgsequence.svg"), "Export frame sequence as PNG-sequence...",
    [this] { exportSequenceAsPNG(false); });
  addAction(act);
  act=fileMenu->addAction(Utils::QIconCached("exportimgsequence.svg"), "Export frame sequence as Video...",
    [this] { exportSequenceAsPNG(true); });
  addAction(act); // must work also if menu bar is invisible
  fileMenu->addAction(Utils::QIconCached("exportiv.svg"), "Export current frame as IV...", this, &MainWindow::exportCurrentAsIV);
  fileMenu->addAction(Utils::QIconCached("exportiv.svg"), "Export current frame as PS...", this, &MainWindow::exportCurrentAsPS);
  fileMenu->addSeparator();
  fileMenu->addAction(Utils::QIconCached("loadwst.svg"), "Load window state...", this, static_cast<void(MainWindow::*)()>(&MainWindow::loadWindowState));
  act=fileMenu->addAction(Utils::QIconCached("savewst.svg"), "Save window state...", this, &MainWindow::saveWindowState);
  addAction(act); // must work also if menu bar is invisible
  act->setShortcut(QKeySequence("Ctrl+W"));
  fileMenu->addAction(Utils::QIconCached("loadcamera.svg"), "Load camera...", this, static_cast<void(MainWindow::*)()>(&MainWindow::loadCamera));
  act=fileMenu->addAction(Utils::QIconCached("savecamera.svg"), "Save camera...", this, &MainWindow::saveCamera);
  act->setShortcutContext(Qt::WidgetWithChildrenShortcut);
  addAction(act); // must work also if menu bar is invisible
  fileMenu->addSeparator();
  act=fileMenu->addAction(Utils::QIconCached("settings.svg"), "Settings...", [this](){
    static QDialog *dialog=nullptr;
    if(dialog==nullptr)
      dialog=new SettingsDialog(this);
    dialog->show();
  });
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
  auto *animationMenu=new QMenu("Animation", menuBar());
  animationMenu->addAction(stopAct);
  animationMenu->addAction(lastFrameAct);
  animationMenu->addAction(playAct);
  menuBar()->addMenu(animationMenu);
  connect(stopAct, &QAction::triggered, this, &MainWindow::stopSCSlot);
  connect(lastFrameAct, &QAction::triggered, this, &MainWindow::lastFrameSCSlot);
  connect(playAct, &QAction::triggered, this, &MainWindow::playSCSlot);


  // scene view menu
  auto *sceneViewMenu=new QMenu("Scene View", menuBar());
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
  QMenu *rotateView=sceneViewMenu->addMenu(Utils::QIconCached("rotateview.svg"),"Rotate view");
  act=rotateView->addAction("About World-X-Axis", this, &MainWindow::viewRotateXpWorld, QKeySequence("X"));
  addAction(act);
  act=rotateView->addAction("About World-X-Axis", this, &MainWindow::viewRotateXmWorld, QKeySequence("Shift+X"));
  addAction(act);
  act=rotateView->addAction("About World-Y-Axis", this, &MainWindow::viewRotateYpWorld, QKeySequence("Y"));
  addAction(act);
  act=rotateView->addAction("About World-Y-Axis", this, &MainWindow::viewRotateYmWorld, QKeySequence("Shift+Y"));
  addAction(act);
  act=rotateView->addAction("About World-Z-Axis", this, &MainWindow::viewRotateZpWorld, QKeySequence("Z"));
  addAction(act);
  act=rotateView->addAction("About World-Z-Axis", this, &MainWindow::viewRotateZmWorld, QKeySequence("Shift+Z"));
  addAction(act);
  rotateView->addSeparator();
  act=rotateView->addAction("About Screen-X-Axis", this, &MainWindow::viewRotateXpScreen, QKeySequence("Ctrl+X"));
  act->setShortcutContext(Qt::WidgetWithChildrenShortcut);
  addAction(act);
  act=rotateView->addAction("About Screen-X-Axis", this, &MainWindow::viewRotateXmScreen, QKeySequence("Ctrl+Shift+X"));
  addAction(act);
  act=rotateView->addAction("About Screen-Y-Axis", this, &MainWindow::viewRotateYpScreen, QKeySequence("Ctrl+Y"));
  addAction(act);
  act=rotateView->addAction("About Screen-Y-Axis", this, &MainWindow::viewRotateYmScreen, QKeySequence("Ctrl+Shift+Y"));
  addAction(act);
  act=rotateView->addAction("About Screen-Z-Axis", this, &MainWindow::viewRotateZpScreen, QKeySequence("Ctrl+Z"));
  act->setShortcutContext(Qt::WidgetWithChildrenShortcut);
  addAction(act);
  act=rotateView->addAction("About Screen-Z-Axis", this, &MainWindow::viewRotateZmScreen, QKeySequence("Ctrl+Shift+Z"));
  act->setShortcutContext(Qt::WidgetWithChildrenShortcut);
  addAction(act);
  sceneViewMenu->addSeparator();
  act=sceneViewMenu->addAction(Utils::QIconCached("frame.svg"),"World frame", this, &MainWindow::showWorldFrameSlot, QKeySequence("W"));
  act->setCheckable(true);
  sceneViewMenu->addSeparator();
  cameraAct=sceneViewMenu->addAction(Utils::QIconCached("camera.svg"),"Toggle camera type", [this](bool checked){
    setCameraType(checked ? SoOrthographicCamera::getClassTypeId() : SoPerspectiveCamera::getClassTypeId());
  }, QKeySequence("C"));
  cameraAct->setCheckable(true);
  cameraAct->setChecked(appSettings->get<int>(AppSettings::cameraType)==0);
  addAction(cameraAct); // must work also if menu bar is invisible

  viewStereo=sceneViewMenu->addAction(Utils::QIconCached("camerastereo.svg"),"Stereo view", [this](bool checked){
    reinit3DView(checked ? StereoType::LeftRight : StereoType::None);
  }, QKeySequence("V"));
  viewStereo->setCheckable(true);
  viewStereo->setChecked(appSettings->get<int>(AppSettings::stereoType)!=static_cast<int>(StereoType::None));
  addAction(viewStereo); // must work also if menu bar is invisible

  sceneViewMenu->addAction(Utils::QIconCached("camerabody.svg"),"Release camera from move with body", this, &MainWindow::releaseCameraFromBodySlot);
  sceneViewMenu->addSeparator();
  engDrawingView=sceneViewMenu->addAction(Utils::QIconCached("engdrawing.svg"),"Engineering drawing", this, &MainWindow::toggleEngDrawingViewSlot);
  engDrawingView->setToolTip("NOTE: If getting unchecked, the outlines of all bodies will be enabled and the shilouette edges are disabled!");
  engDrawingView->setStatusTip(engDrawingView->toolTip());
  engDrawingView->setCheckable(true);
  menuBar()->addMenu(sceneViewMenu);

  // gui view menu
  auto *guiViewMenu=new QMenu("GUI View", menuBar());
  toggleMenuBar=guiViewMenu->addAction("Menu bar", this, &MainWindow::toggleMenuBarSlot, QKeySequence(Qt::Key_F10));
  addAction(toggleMenuBar); // must work also if menu bar is invisible
  toggleMenuBar->setCheckable(true);
  toggleMenuBar->setChecked(true);
  toggleStatusBar=guiViewMenu->addAction("Status bar", this, &MainWindow::toggleStatusBarSlot);
  toggleStatusBar->setCheckable(true);
  toggleStatusBar->setChecked(true);
  toggleFrameSlider=guiViewMenu->addAction("Frame/Time slider", this, &MainWindow::toggleFrameSliderSlot);
  toggleFrameSlider->setCheckable(true);
  toggleFrameSlider->setChecked(true);
  QAction *toggleFullScreen=guiViewMenu->addAction("Full screen", this, &MainWindow::toggleFullScreenSlot, QKeySequence(Qt::Key_F5));
  addAction(toggleFullScreen); // must work also if menu bar is invisible
  toggleFullScreen->setCheckable(true);
  toggleDecoration=guiViewMenu->addAction("Window decoration", this, &MainWindow::toggleDecorationSlot);
  toggleDecoration->setCheckable(true);
  toggleDecoration->setChecked(true);
  menuBar()->addMenu(guiViewMenu);

  // dock menu
  auto *dockMenu=new QMenu("Docks", menuBar());
  dockMenu->addAction(objectListDW->toggleViewAction());
  dockMenu->addAction(objectInfoDW->toggleViewAction());
  menuBar()->addMenu(dockMenu);

  // file toolbar
  auto *fileTB=new QToolBar("File toolbar", this);
  fileTB->setObjectName("MainWindow::fileTB");
  addToolBar(Qt::TopToolBarArea, fileTB);
  fileTB->addAction(addFileAct);

  // view toolbar
  auto *viewTB=new QToolBar("View toolbar", this);
  viewTB->setObjectName("MainWindow::viewTB");
  addToolBar(Qt::TopToolBarArea, viewTB);
  viewTBActions.emplace_back(viewAllAct);
  viewTBActions.emplace_back(nullptr);
  viewTBActions.emplace_back(topViewAct);
  viewTBActions.emplace_back(bottomViewAct);
  viewTBActions.emplace_back(frontViewAct);
  viewTBActions.emplace_back(backViewAct);
  viewTBActions.emplace_back(rightViewAct);
  viewTBActions.emplace_back(leftViewAct);
  viewTBActions.emplace_back(nullptr);
  viewTBActions.emplace_back(isometriViewAct);
  viewTBActions.emplace_back(dimetricViewAct);
  viewTBActions.emplace_back(nullptr);
  viewTBActions.emplace_back(cameraAct);
  for(auto *a : viewTBActions)
    if(a)
      viewTB->addAction(a);
    else
      viewTB->addSeparator();

  // animation toolbar
  auto *animationTB=new QToolBar("Animation toolbar", this);
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
  speedSB->setMinimum(0);
  speedSB->setMaximum(1e6);
  speedSB->setSingleStep(1e-1);
  speedSB->setMaximumSize(65, 1000);
  speedSB->setDecimals(6);
  speedSB->setButtonSymbols(QDoubleSpinBox::NoButtons);
  speedSB->setValue(1.0);
  connect(speedSB, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &MainWindow::restartPlay);
  auto *speedWG=new QWidget(this);
  auto *speedLO=new QGridLayout(speedWG);
  speedLO->setSpacing(0);
  speedLO->setContentsMargins(0,0,0,0);
  speedWG->setLayout(speedLO);
  auto *speedL=new QLabel("Speed:", this);
  speedLO->addWidget(speedL, 0, 0);
  speedLO->addWidget(speedSB, 1, 0);
  speedWheel=new QwtWheel(this);
  speedWheel->setWheelWidth(15);
  connect(speedWheel, SIGNAL(valueChanged(double)), this, SLOT(speedWheelChangedD(double))); // using function pointers is not working on Windows across DLLs
  connect(speedWheel, SIGNAL(wheelPressed()), this, SLOT(speedWheelPressed())); // using function pointers is not working on Windows across DLLs
  connect(speedWheel, SIGNAL(wheelReleased()), this, SLOT(speedWheelReleased())); // using function pointers is not working on Windows across DLLs
  speedWheel->setRange(-10000, 10000);
  speedWheel->setOrientation(Qt::Vertical);
  speedLO->addWidget(speedWheel, 0, 1, 2, 1);
  animationTB->addWidget(speedWG);
  connect(new QShortcut(QKeySequence(Qt::Key_PageUp),this), &QShortcut::activated, this, &MainWindow::speedUpSlot);
  connect(new QShortcut(QKeySequence(Qt::Key_PageDown),this), &QShortcut::activated, this, &MainWindow::speedDownSlot);
  animationTB->addSeparator();
  // frame spin box
  frameSB=new QSpinBox;
  frameSB->setMinimumSize(55,0);
  auto *frameWG=new QWidget(this);
  auto *frameLO=new QGridLayout(frameWG);
  frameLO->setSpacing(0);
  frameLO->setContentsMargins(0,0,0,0);
  frameWG->setLayout(frameLO);
  auto *frameL=new QLabel("Frame:", this);
  frameLO->addWidget(frameL, 0, 0);
  frameLO->addWidget(frameSB, 1, 0);
  animationTB->addWidget(frameWG);
  connect(frameSB, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &MainWindow::updateFrame);
  connect(timeSlider, &QTripleSlider::currentRangeChanged, this, &MainWindow::frameSBSetRange);
  connect(timeSlider, &QTripleSlider::currentRangeChanged, this, &MainWindow::restartPlay);
  connect(timeSlider, &QTripleSlider::currentRangeChanged, this, &MainWindow::frameMinMaxSetValue);
  connect(new QShortcut(QKeySequence(Qt::Key_Up),this), &QShortcut::activated, frameSB, &QSpinBox::stepUp);
  connect(new QShortcut(QKeySequence(Qt::Key_Down),this), &QShortcut::activated, frameSB, &QSpinBox::stepDown);
  connect(new QShortcut(QKeySequence(Qt::Key_K),this), &QShortcut::activated, frameSB, &QSpinBox::stepUp);
  connect(new QShortcut(QKeySequence(Qt::Key_J),this), &QShortcut::activated, frameSB, &QSpinBox::stepDown);
  // min frame spin box
  frameMinSB=new QSpinBox;
  frameMinSB->setMinimumSize(55,0);
  frameMinSB->setRange(0, 0);
  auto *frameMinWG=new QWidget(this);
  auto *frameMinLO=new QGridLayout(frameMinWG);
  frameMinLO->setSpacing(0);
  frameMinLO->setContentsMargins(0,0,0,0);
  frameMinWG->setLayout(frameMinLO);
  auto *frameMinL=new QLabel("Min:", this);
  frameMinLO->addWidget(frameMinL, 0, 0);
  frameMinLO->addWidget(frameMinSB, 1, 0);
  animationTB->addWidget(frameMinWG);
  connect(frameMinSB, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), timeSlider, &QTripleSlider::setCurrentMinimum);
  // max frame spin box
  frameMaxSB=new QSpinBox;
  frameMaxSB->setMinimumSize(55,0);
  frameMaxSB->setRange(0, 0);
  auto *frameMaxWG=new QWidget(this);
  auto *frameMaxLO=new QGridLayout(frameMaxWG);
  frameMaxLO->setSpacing(0);
  frameMaxLO->setContentsMargins(0,0,0,0);
  frameMaxWG->setLayout(frameMaxLO);
  auto *frameMaxL=new QLabel("Max:", this);
  frameMaxLO->addWidget(frameMaxL, 0, 0);
  frameMaxLO->addWidget(frameMaxSB, 1, 0);
  animationTB->addWidget(frameMaxWG);
  connect(frameMaxSB, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), timeSlider, &QTripleSlider::setCurrentMaximum);

  // tool menu
  auto *toolMenu=new QMenu("Tools", menuBar());
  toolMenu->addAction(fileTB->toggleViewAction());
  toolMenu->addAction(viewTB->toggleViewAction());
  toolMenu->addAction(animationTB->toggleViewAction());
  menuBar()->addMenu(toolMenu);

  // help menu
  menuBar()->addSeparator();
  auto *helpMenu=new QMenu("Help", menuBar());
  helpMenu->addAction(Utils::QIconCached("help.svg"), "GUI help...", this, &MainWindow::guiHelp);
  helpMenu->addAction(Utils::QIconCached("help.svg"), "XML help...", this, &MainWindow::xmlHelp);
  helpMenu->addAction(Utils::QIconCached((installPath/"share"/"openmbv"/"icons"/"openmbv.svg").string().c_str()),
                      "About OpenMBV...", this, &MainWindow::aboutOpenMBV);
  menuBar()->addMenu(helpMenu);

  // status bar
  auto *sb=new QStatusBar(this);
  fps=new QLabel("FPS: -");
  fpsTime=new QElapsedTimer();
  sb->addPermanentWidget(fps);
  setStatusBar(sb);

  // register callback function on frame change
  frameSensor=new SoFieldSensor(frameSensorCB, this);
  frameSensor->attach(frame);

  hdf5RefreshDelta=appSettings->get<int>(AppSettings::hdf5RefreshDelta);

  // animation timer
  animTimer=new QTimer(this);
  connect(animTimer, &QTimer::timeout, this, &MainWindow::heavyWorkSlot);
  time=new QElapsedTimer();
  hdf5RefreshTimer=new QTimer(this);
  connect(hdf5RefreshTimer, &QTimer::timeout, this, &MainWindow::hdf5RefreshSlot);
  if(hdf5RefreshDelta>0)
    hdf5RefreshTimer->start(hdf5RefreshDelta);

  // react on parameters

  // line width for outline and shilouette edges
  olseDrawStyle=new SoDrawStyle;
  olseDrawStyle->ref();
  olseDrawStyle->style.setValue(SoDrawStyle::LINES);
  olseDrawStyle->lineWidth.setValue(1);

  // boundingbox/highlight color
  for(auto &[color,colorAS,drawStyle,linewidthAS] : std::array<tuple<SoBaseColor*&, AppSettings::AS, SoDrawStyle*&, AppSettings::AS>,2>{{
      {bboxColor     , AppSettings::boundingBoxLineColor, bboxDrawStyle     , AppSettings::boundingBoxLineWidth},
      {highlightColor, AppSettings::highlightLineColor  , highlightDrawStyle, AppSettings::highlightLineWidth}}}) {
    color=new SoBaseColor;
    color->ref();
    auto rgb=appSettings->get<QColor>(colorAS).rgb();
    color->rgb.setValue(qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
    drawStyle=new SoDrawStyle;
    drawStyle->ref();
    drawStyle->style.setValue(SoDrawStyle::LINES);
    drawStyle->lineWidth.setValue(appSettings->get<double>(linewidthAS));
  }

  // complexity
  complexity->type.setValue(SoComplexity::SCREEN_SPACE);
  complexity->value.setValue(0.2);

  // color for outline and shilouette edges
  olseColor=new SoBaseColorHeavyOverride;
  olseColor->ref();
  olseColor->rgb.set1Value(0, 0,0,0);

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
    if(SoBase::read(&input, newHeadLight, SoVRMLDirectionalLight::getClassTypeId())) {
      glViewer->getHeadlight()->on.setValue(((SoVRMLDirectionalLight*)newHeadLight)->on.getValue());
      glViewer->getHeadlight()->intensity.setValue(((SoVRMLDirectionalLight*)newHeadLight)->intensity.getValue());
      glViewer->getHeadlight()->color.setValue(((SoVRMLDirectionalLight*)newHeadLight)->color.getValue());
      glViewer->getHeadlight()->direction.setValue(((SoVRMLDirectionalLight*)newHeadLight)->direction.getValue());
    }
    arg.erase(i); arg.erase(i2);
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
  viewAllSlot();

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
    connect(timer, &QTimer::timeout,this, [this, timer](){
      if(waitFor.empty()) {
        timer->stop();
        if(!close())
          timer->start(100);
      }
    });
    timer->start(100);
  }

  // apply settings
  {
    QTapAndHoldGesture::setTimeout(appSettings->get<int>(AppSettings::tapAndHoldTimeout));
    olseDrawStyle->lineWidth.setValue(appSettings->get<double>(AppSettings::outlineShilouetteEdgeLineWidth));
    auto color=appSettings->get<QColor>(AppSettings::outlineShilouetteEdgeLineColor);
    auto rgb=color.rgb();
    olseColor->rgb.set1Value(0, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
    int complexityType=appSettings->get<int>(AppSettings::complexityType);
    complexity->type.setValue( complexityType==0 ? SoComplexity::OBJECT_SPACE :
                              (complexityType==1 ? SoComplexity::SCREEN_SPACE :
                                                   SoComplexity::BOUNDING_BOX));
    complexity->value.setValue(appSettings->get<double>(AppSettings::complexityValue));
    color=appSettings->get<QColor>(AppSettings::topBackgroudColor);
    rgb=color.rgb();
    bgColor->set1Value(2, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
    bgColor->set1Value(3, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
    fgColorTop->set1Value(0, 1-(color.value()+127)/255,1-(color.value()+127)/255,1-(color.value()+127)/255);
    color=appSettings->get<QColor>(AppSettings::bottomBackgroundColor);
    rgb=color.rgb();
    bgColor->set1Value(0, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
    bgColor->set1Value(1, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
    fgColorBottom->set1Value(0, 1-(color.value()+127)/255,1-(color.value()+127)/255,1-(color.value()+127)/255);

    setStereoOffset(appSettings->get<double>(AppSettings::stereoOffset));
  }

  reinit3DView(static_cast<StereoType>(appSettings->get<int>(AppSettings::stereoType)));
}

class DialogStereo : public QDialog {
  public:
    DialogStereo();
    ~DialogStereo() override;
    void closeEvent(QCloseEvent *event) override;
    void showEvent(QShowEvent *event) override;
  private:
    QPushButton *fullScreenButton;
};

DialogStereo::DialogStereo() {
  static const boost::filesystem::path installPath(boost::dll::program_location().parent_path().parent_path());

  setWindowFlags(Qt::Window | Qt::CustomizeWindowHint | Qt::WindowTitleHint);
  setWindowTitle("OpenMBV - Stereo View");
  setWindowIcon(Utils::QIconCached((installPath/"share"/"openmbv"/"icons"/"openmbv.svg").string()));
  connect(new QShortcut(QKeySequence("F5"),this), &QShortcut::activated, [this](){
    if(isFullScreen()) {
      showNormal();
      fullScreenButton->show();
    }
    else {
      showFullScreen();
      fullScreenButton->hide();
    }
  });
  connect(new QShortcut(QKeySequence("ESC"),this), &QShortcut::activated, [this](){
    showNormal();
    fullScreenButton->show();
  });
  connect(new QShortcut(QKeySequence("V"),this), &QShortcut::activated, [](){
    MainWindow::getInstance()->reinit3DView(MainWindow::StereoType::None);
  });

  auto *mw=MainWindow::getInstance();

  auto *dialogStereoLO=new QGridLayout(this);
  dialogStereoLO->setContentsMargins(0,0,0,0);
  dialogStereoLO->setSpacing(0);
  setLayout(dialogStereoLO);

  dialogStereoLO->addWidget(mw->glViewerWG, 0,0);
  mw->cameraAct->setEnabled(false);
  mw->cameraAct->setChecked(false);
  mw->setCameraType(SoPerspectiveCamera::getClassTypeId());
  mw->glViewer->getCamera()->setStereoMode(SoCamera::LEFT_VIEW);
  mw->glViewer->getCamera()->viewportMapping.setValue(SoCamera::LEAVE_ALONE);
  mw->glViewer->setAspectRatio(2.0);

  auto *glViewerWGRight=new MyTouchWidget(this);
  fullScreenButton=new QPushButton(Utils::QIconCached("fullscreen.svg"), "", this);
  fullScreenButton->setIconSize(QSize(50,50));
  fullScreenButton->setFixedSize(QSize(60,60));
  QTimer::singleShot(0, [this](){ // isFullScreen is not working inside of the ctor (returns always false)
    if(isFullScreen())
      fullScreenButton->hide();
    else
      fullScreenButton->show();
  });
  connect(fullScreenButton, &QPushButton::clicked, [this](){
    showFullScreen();
    fullScreenButton->hide();
  });
  mw->glViewerRight=new SoQtMyViewer(glViewerWGRight, mw->transparency);
  mw->glViewerRight->setSceneGraph(mw->sceneRoot);
  auto *camera=static_cast<SoPerspectiveCamera*>(mw->glViewer->getCamera());
  dialogStereoLO->addWidget(glViewerWGRight, 0,1);
  mw->glViewerRight->setCameraType(SoPerspectiveCamera::getClassTypeId());
  auto *cameraRight=static_cast<SoPerspectiveCamera*>(mw->glViewerRight->getCamera());
  cameraRight->setStereoMode(SoCamera::RIGHT_VIEW);
  mw->glViewerRight->getCamera()->viewportMapping.setValue(SoCamera::LEAVE_ALONE);
  mw->glViewerRight->setAspectRatio(2.0);
  mw->glViewerRight->setStereoOffset(appSettings->get<double>(AppSettings::stereoOffset));

  auto *positionE=new SoGate(SoMFVec3f::getClassTypeId());
  positionE->enable=true;
  positionE->input->connectFrom(&camera->position);
  cameraRight->position.connectFrom(positionE->output);
  
  auto *rotationE=new SoGate(SoMFRotation::getClassTypeId());
  rotationE->enable=true;
  rotationE->input->connectFrom(&camera->orientation);
  cameraRight->orientation.connectFrom(rotationE->output);
  
  auto *focalDistanceE=new SoGate(SoMFFloat::getClassTypeId());
  focalDistanceE->enable=true;
  focalDistanceE->input->connectFrom(&camera->focalDistance);
  cameraRight->focalDistance.connectFrom(focalDistanceE->output);
  
  auto *heightE=new SoGate(SoMFFloat::getClassTypeId());
  heightE->enable=true;
  heightE->input->connectFrom(&camera->heightAngle);
  cameraRight->heightAngle.connectFrom(heightE->output);
}

DialogStereo::~DialogStereo() {
  // SoQtViewer does not delete it self if its parent is deleted -> delete it explicitly
  delete MainWindow::getInstance()->glViewerRight;
  MainWindow::getInstance()->glViewerRight=nullptr;
}

void DialogStereo::closeEvent(QCloseEvent *event) {
  appSettings->set(AppSettings::dialogstereo_geometry, saveGeometry());
  QDialog::closeEvent(event);
}

void DialogStereo::showEvent(QShowEvent *event) {
  auto data=appSettings->get<QByteArray>(AppSettings::dialogstereo_geometry);
  if(!data.isEmpty())
    restoreGeometry(data);
  else
    setGeometry(50,50,750,550);
  QDialog::showEvent(event);
}

void MainWindow::reinit3DView(StereoType stereoType) {
  appSettings->set(AppSettings::stereoType, static_cast<int>(stereoType));
  viewStereo->setChecked(stereoType!=StereoType::None);
  switch(stereoType) {
    case StereoType::None:
      mainLO->addWidget(glViewerWG,0,0);
      if(dialogStereo) dialogStereo->close();
      delete dialogStereo;
      dialogStereo=nullptr;
      cameraAct->setEnabled(true);
      cameraAct->setChecked(appSettings->get<int>(AppSettings::cameraType)==0);
      setCameraType(appSettings->get<int>(AppSettings::cameraType)==1 ?
                    SoPerspectiveCamera::getClassTypeId() : SoOrthographicCamera::getClassTypeId());
      glViewer->getCamera()->setStereoMode(SoCamera::MONOSCOPIC);
      glViewer->getCamera()->viewportMapping.setValue(SoCamera::ADJUST_CAMERA);
      glViewer->setAspectRatio(1.0);
      break;
    case StereoType::LeftRight:
      auto *disableStereo=new QPushButton(Utils::QIconCached("camerastereodisable.svg"), "", this);
      disableStereo->setIconSize(QSize(100,100));
      connect(disableStereo, &QPushButton::clicked, [this](){
        reinit3DView(StereoType::None);
      });
      mainLO->addWidget(disableStereo,0,0);
      if(dialogStereo) dialogStereo->close();
      delete dialogStereo;
      dialogStereo=new DialogStereo;
      dialogStereo->show();
    break;
  }
}

void MainWindow::setStereoOffset(double value) {
  glViewer->setStereoOffset(value);
  if(glViewerRight) glViewerRight->setStereoOffset(value);
}

MainWindow* const MainWindow::getInstance() {
  return instance;
}

void MainWindow::highlightObject(Object *current) {
  // disable all highlights
  Utils::visitTreeWidgetItems<Object*>(objectList->invisibleRootItem(), [](Object *obj) {
    obj->setHighlight(false);
  });
  // enable current highlights
  if(current)
    current->setHighlight(true);
}

void MainWindow::highlightObject(const string &curID) {
  // disable all highlight
  Utils::visitTreeWidgetItems<Object*>(objectList->invisibleRootItem(), [](Object *obj) {
    obj->setHighlight(false);
  });
  // enable all curID highlights
  if(!curID.empty())
    Utils::visitTreeWidgetItems<Object*>(objectList->invisibleRootItem(), [curID](auto && obj) {
      if(obj->object->getID()!=curID)
        return;
      obj->setHighlight(true);
    });
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
  bboxColor->unref();
  bboxDrawStyle->unref();
  highlightColor->unref();
  highlightDrawStyle->unref();
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
  OpenMBVGUI::appSettings.reset();
}

bool MainWindow::openFile(const std::string& fileName, QTreeWidgetItem* parentItem, SoGroup *soParent, int ind) {
  fmatvec::AdoptCurrentMessageStreamsUntilScopeExit dummy(this);

  // default parameter
  if(parentItem==nullptr) parentItem=objectList->invisibleRootItem();
  if(soParent==nullptr) soParent=sceneRoot;

  // read XML
  std::shared_ptr<OpenMBV::Group> rootGroup=OpenMBV::ObjectFactory::create<OpenMBV::Group>();
  rootGroup->setFileName(fileName);
  {
    // lock mutex to avoid that the callback from rootGroup->read(...) is called before the rootGroupOMBV is build
    std::scoped_lock lock(mutex);
    std::shared_ptr<Group*> rootGroupOMBV(new Group*);
    rootGroup->setCloseRequestCallback([this, rootGroupOMBV](){
      // lock mutex to avoid that this callback tries to acces rootGroupOMBV before it is set
      std::scoped_lock lock(mutex);
      // only call signals here since this is executed in a different thread
      (*rootGroupOMBV)->reloadFileSignal();
    });
    rootGroup->setRefreshCallback([this, rootGroupOMBV](){
      // lock mutex to avoid that this callback tries to acces rootGroupOMBV before it is set
      std::scoped_lock lock(mutex);
      // only call signals here since this is executed in a different thread
      (*rootGroupOMBV)->refreshFileSignal();
    });
    rootGroup->read();

    // Duplicate OpenMBVCppInterface tree using OpenMBV tree
    (*rootGroupOMBV)=static_cast<Group*>(ObjectFactory::create(rootGroup, parentItem, soParent, ind));
    (*rootGroupOMBV)->setText(0, fileName.c_str());
    (*rootGroupOMBV)->setToolTip(0, QFileInfo(fileName.c_str()).absoluteFilePath());
    (*rootGroupOMBV)->getIconFile()="h5file.svg";
    (*rootGroupOMBV)->setIcon(0, Utils::QIconCached((*rootGroupOMBV)->getIconFile()));
    // the mutex is release now and the callback can deliver rootGroupOMBV->reloadFileSignal() call from now on
  }

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
  viewAllSlot();
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
  viewAllSlot();
}

void MainWindow::toggleAction(Object *current, QAction *currentAct) {
  QList<QAction*> actions=current->getProperties()->getActions();
  for(auto & action : actions)
    if(action->objectName()==currentAct->objectName() && currentAct!=action)
      action->trigger();
}
void MainWindow::execPropertyMenu(const std::vector<QAction*> &additionalActions) {
  auto *object=(Object*)objectList->currentItem();
  if(!object)
    return;
  QMenu* menu=object->getProperties()->getContextMenu();
  for(auto &a: additionalActions)
    menu->addAction(a);
  QAction *currentAct=menu->exec(QCursor::pos());
  for(auto &a: additionalActions) {
    menu->removeAction(a);
    delete a;
  }
  // if action is not NULL and the action has a object name trigger also the actions with
  // the same name of all other selected objects
  if(currentAct && currentAct->objectName()!="")
    Utils::visitTreeWidgetItems<Object*>(objectList->invisibleRootItem(), [currentAct](auto && PH1) { return toggleAction(std::forward<decltype(PH1)>(PH1), currentAct); }, true);
}

void MainWindow::objectListClicked() {
  auto *curObj=static_cast<Object*>(objectList->currentItem());
  objectSelected(curObj ? curObj->object->getID() : "", curObj);
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
    Utils::enableTouch(text);
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
    auto *icon=new QLabel;
    layout->addWidget(icon, 0, 0, Qt::AlignTop);
    QFontInfo fontinfo(font());
    icon->setPixmap(Utils::QIconCached((boost::dll::program_location().parent_path().parent_path()/"share"/"openmbv"/"icons"/"openmbv.svg")
                    .string().c_str()).pixmap(fontinfo.pixelSize()*3,fontinfo.pixelSize()*3));
    auto *text=new QTextEdit;
    layout->addWidget(text, 0, 1);
    Utils::enableTouch(text);
    text->setReadOnly(true);
    text->setHtml(
      "<h1>OpenMBV - Open Multi Body Viewer</h1>"
      "<p>Copyright &copy; Markus Friedrich <tt>&lt;friedrich.at.gc@googlemail.com&gt;</tt><p/>"
      "<p>Licensed under the Lesser General Public License (see file COPYING).</p>"
      "<p>This is free software; see the source for copying conditions.  There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.</p>"
      "<h2>Authors:</h2>"
      "<ul>"
      "  <li>Markus Friedrich <tt>&lt;friedrich.at.gc@googlemail.com&gt;</tt> (Maintainer)</li>"
      "</ul>"
      "<h2>Dependencies:</h2>"
      "<pre>"
#include "../NOTICE"
      "</pre>"
      "<p>A special thanks to all authors of these projects.</p>"
    );
  }
  about->show();
}

void MainWindow::viewChange(ViewSide side) {
  auto rotateTo=[this](const SbRotation &cameraOri, bool noAni=false){
    auto *camera=glViewer->getCamera();
    auto initialCameraOri=camera->orientation.getValue();
    SbMatrix oriMatrix;
    initialCameraOri.getValue(oriMatrix);
    SbVec3f initialCameraVec(oriMatrix[2][0], oriMatrix[2][1], oriMatrix[2][2]);
    auto initialCameraPos=camera->position.getValue();
    auto toPoint=initialCameraPos-initialCameraVec*camera->focalDistance.getValue();
    cameraOri.getValue(oriMatrix);
    SbVec3f cameraVec(oriMatrix[2][0], oriMatrix[2][1], oriMatrix[2][2]);
    auto cameraPos=toPoint+cameraVec*camera->focalDistance.getValue();
    auto relOri=initialCameraOri.inverse()*cameraOri;
    SbVec3f axis;
    float angle;
    relOri.getValue(axis, angle);
    startShortAni([camera, initialCameraPos, cameraPos, initialCameraOri, axis, angle](double c){
      camera->position.setValue(initialCameraPos + (cameraPos-initialCameraPos) * (0.5-0.5*cos(c*M_PI)));
      SbRotation relOri(axis, angle*(0.5-0.5*cos(c*M_PI)));
      camera->orientation.setValue(initialCameraOri*relOri);
    }, noAni);
  };
  SbRotation r, r2;
  SbVec3f n;
  switch(side) {
    case top:
      rotateTo(SbMatrix( 1, 0,0,0,   0,1,0,0,    0, 0, 1,0,   0,0,0,1));
      break;
    case bottom:
      rotateTo(SbMatrix(-1, 0,0,0,   0,1,0,0,    0, 0,-1,0,   0,0,0,1));
      break;
    case front:
      rotateTo(SbMatrix( 1, 0,0,0,   0,0,1,0,    0,-1, 0,0,   0,0,0,1));
      break;
    case back:
      rotateTo(SbMatrix(-1, 0,0,0,   0,0,1,0,    0, 1, 0,0,   0,0,0,1));
      break;
    case right:
      rotateTo(SbMatrix( 0, 1,0,0,   0,0,1,0,    1, 0, 0,0,   0,0,0,1));
      break;
    case left:
      rotateTo(SbMatrix( 0,-1,0,0,   0,0,1,0,   -1, 0, 0,0,   0,0,0,1));
      break;
    case isometric:
      //glViewer->getCamera()->position.setValue(1,1,1);
      //glViewer->getCamera()->pointAt(SbVec3f(0,0,0), SbVec3f(0,0,1));
      rotateTo(SbMatrix(-0.707107,0.707107,0,0,   -0.408248,-0.408248,0.816497,0,   0.57735,0.57735,0.57735,0,   0,0,0,1));
      break;
    case dimetric:
      //glViewer->getCamera()->orientation.setValue(Utils::cardan2Rotation(SbVec3f(-1.227769277146394,0,1.227393504015536)));
      rotateTo(SbMatrix(0.336693,-0.941614,0,0,   0.316702,0.113243,0.941741,0,   -0.886757,-0.317078,0.336339,0,   0,0,0,1));
      break;
    case rotateXpWorld:
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(SbVec3f(-1,0,0), appSettings->get<double>(AppSettings::anglePerKeyPress)*M_PI/180);
      rotateTo(r, true);
      break;
    case rotateXmWorld:
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(SbVec3f(-1,0,0), -appSettings->get<double>(AppSettings::anglePerKeyPress)*M_PI/180);
      rotateTo(r, true);
      break;
    case rotateYpWorld:
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(SbVec3f(0,-1,0), appSettings->get<double>(AppSettings::anglePerKeyPress)*M_PI/180);
      rotateTo(r, true);
      break;
    case rotateYmWorld:
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(SbVec3f(0,-1,0), -appSettings->get<double>(AppSettings::anglePerKeyPress)*M_PI/180);
      rotateTo(r, true);
      break;
    case rotateZpWorld:
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(SbVec3f(0,0,-1), appSettings->get<double>(AppSettings::anglePerKeyPress)*M_PI/180);
      rotateTo(r, true);
      break;
    case rotateZmWorld:
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(SbVec3f(0,0,-1), -appSettings->get<double>(AppSettings::anglePerKeyPress)*M_PI/180);
      rotateTo(r, true);
      break;
    case rotateXpScreen:
      r2=glViewer->getCamera()->orientation.getValue(); // camera orientation
      r2*=((SoSFRotation*)(cameraOrientation->outRotation[0]))->getValue(); // camera orientation relative to "Move Camera with Body"
      r2.multVec(SbVec3f(-1,0,0),n);
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(n, appSettings->get<double>(AppSettings::anglePerKeyPress)*M_PI/180);
      rotateTo(r, true);
      break;
    case rotateXmScreen:
      r2=glViewer->getCamera()->orientation.getValue(); // camera orientation
      r2*=((SoSFRotation*)(cameraOrientation->outRotation[0]))->getValue(); // camera orientation relative to "Move Camera with Body"
      r2.multVec(SbVec3f(-1,0,0),n);
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(n, -appSettings->get<double>(AppSettings::anglePerKeyPress)*M_PI/180);
      rotateTo(r, true);
      break;
    case rotateYpScreen:
      r2=glViewer->getCamera()->orientation.getValue(); // camera orientation
      r2*=((SoSFRotation*)(cameraOrientation->outRotation[0]))->getValue(); // camera orientation relative to "Move Camera with Body"
      r2.multVec(SbVec3f(0,-1,0),n);
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(n, appSettings->get<double>(AppSettings::anglePerKeyPress)*M_PI/180);
      rotateTo(r, true);
      break;
    case rotateYmScreen:
      r2=glViewer->getCamera()->orientation.getValue(); // camera orientation
      r2*=((SoSFRotation*)(cameraOrientation->outRotation[0]))->getValue(); // camera orientation relative to "Move Camera with Body"
      r2.multVec(SbVec3f(0,-1,0),n);
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(n, -appSettings->get<double>(AppSettings::anglePerKeyPress)*M_PI/180);
      rotateTo(r, true);
      break;
    case rotateZpScreen:
      r2=glViewer->getCamera()->orientation.getValue(); // camera orientation
      r2*=((SoSFRotation*)(cameraOrientation->outRotation[0]))->getValue(); // camera orientation relative to "Move Camera with Body"
      r2.multVec(SbVec3f(0,0,-1),n);
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(n, appSettings->get<double>(AppSettings::anglePerKeyPress)*M_PI/180);
      rotateTo(r, true);
      break;
    case rotateZmScreen:
      r2=glViewer->getCamera()->orientation.getValue(); // camera orientation
      r2*=((SoSFRotation*)(cameraOrientation->outRotation[0]))->getValue(); // camera orientation relative to "Move Camera with Body"
      r2.multVec(SbVec3f(0,0,-1),n);
      r=glViewer->getCamera()->orientation.getValue();
      r*=SbRotation(n, -appSettings->get<double>(AppSettings::anglePerKeyPress)*M_PI/180);
      rotateTo(r, true);
      break;
  }
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
    // request a flush of all writers
    requestHDF5Flush();
    // get number of rows of first none enviroment body
    if(!openMBVBodyForLastFrame) {
      auto it=Body::getBodyMap().begin();
      while(it!=Body::getBodyMap().end() && std::static_pointer_cast<OpenMBV::Body>(it->second->object)->getRows()==0)
        it++;
      if(it==Body::getBodyMap().end())
        return;
      openMBVBodyForLastFrame=std::static_pointer_cast<OpenMBV::Body>(it->second->object);
    }
    // use number of rows for found first none enviroment body
    int currentNumOfRows=openMBVBodyForLastFrame->getRows();
    if(deltaTime==0 && currentNumOfRows>=2)
      deltaTime=openMBVBodyForLastFrame->getRow(1)[0]-openMBVBodyForLastFrame->getRow(0)[0];
    if(currentNumOfRows==0) return;

    // update if a new row is available
    if(currentNumOfRows-1!=timeSlider->totalMaximum() || currentNumOfRows-1!=static_cast<int>(frame->getValue())) {
      timeSlider->setTotalMaximum(currentNumOfRows-1);
      timeSlider->setCurrentMaximum(currentNumOfRows-1);
      frame->setValue(currentNumOfRows-1);
    }
  }
}

void MainWindow::hdf5RefreshSlot() {
  // request a flush of all writers
  requestHDF5Flush();
  // get number of rows of first none enviroment body
  if(!openMBVBodyForLastFrame) {
    auto it=Body::getBodyMap().begin();
    while(it!=Body::getBodyMap().end() && std::static_pointer_cast<OpenMBV::Body>(it->second->object)->getRows()==0)
      it++;
    if(it==Body::getBodyMap().end())
      return;
    openMBVBodyForLastFrame=std::static_pointer_cast<OpenMBV::Body>(it->second->object);
  }
  // use number of rows for found first none enviroment body
  int currentNumOfRows=openMBVBodyForLastFrame->getRows();
  if(deltaTime==0 && currentNumOfRows>=2)
    deltaTime=openMBVBodyForLastFrame->getRow(1)[0]-openMBVBodyForLastFrame->getRow(0)[0];
  // update if a the number of rows has changed
  if(currentNumOfRows-1!=timeSlider->totalMaximum()) {
    timeSlider->setTotalMaximum(currentNumOfRows-1);
    if(timeSlider->currentMaximum()>currentNumOfRows-1)
      timeSlider->setCurrentMaximum(currentNumOfRows-1);
    if(static_cast<int>(frame->getValue())>currentNumOfRows-1)
      frame->setValue(currentNumOfRows-1);
    restartPlay();
  }
}

void MainWindow::requestHDF5Flush() {
  for(int i=0; i<objectList->topLevelItemCount(); ++i) {
    auto grp=static_cast<Group*>(objectList->topLevelItem(i));
    grp->requestFlush();
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
  ExportDialog dialog(this, false, false);
  dialog.exec();
  if(dialog.result()==QDialog::Rejected) return;

  QString str("Exporting current frame to %1, please wait!");
  str=str.arg(QFileInfo(dialog.getFileName()).absoluteFilePath());
  statusBar()->showMessage(str);
  msg(Info)<<str.toStdString()<<endl;
  SbVec2s size=glViewer->getSceneManager()->getViewportRegion().getWindowSize()*dialog.getScale();
  short width, height; size.getValue(width, height);
  glViewer->font->size.setValue(glViewer->font->size.getValue()*dialog.getScale());
  exportAsPNG(width, height, dialog.getFileName().toStdString(), dialog.getTransparent());
  glViewer->font->size.setValue(glViewer->font->size.getValue()/dialog.getScale());
}

void MainWindow::exportSequenceAsPNG(bool video) {
  ExportDialog dialog(this, true, video);
  dialog.exec();
  if(dialog.result()==QDialog::Rejected) return;
  double scale=dialog.getScale();
  bool transparent=dialog.getTransparent();
  QString fileName=dialog.getFileName();
  if(!video && !fileName.toUpper().endsWith(".PNG")) return;
  QString pngBaseName=fileName;
  unique_ptr<QTemporaryDir> tmpDir;
  if(!video)
    pngBaseName=pngBaseName.remove(pngBaseName.length()-4,4);
  else {
    tmpDir=std::make_unique<QTemporaryDir>();
    pngBaseName=tmpDir->filePath("openmbv");
  }
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
  glViewer->font->size.setValue(glViewer->font->size.getValue()*scale);
  for(int frame_=startFrame; frame_<=endFrame; frame_=(int)(speed/deltaTime/fps*++videoFrame+startFrame)) {
    QString str("Exporting frame sequence to %1_<nr>.png, please wait! (%2\%)");
    str=str.arg(QFileInfo(pngBaseName).absoluteFilePath()).arg(100.0*videoFrame/lastVideoFrame,0,'f',1);
    statusBar()->showMessage(str);
    msg(Info)<<str.toStdString()<<endl;
    frame->setValue(frame_);
    exportAsPNG(width, height, QString("%1_%2.png").arg(pngBaseName).arg(videoFrame, 6, 10, QChar('0')).toStdString(), transparent);
  }
  glViewer->font->size.setValue(glViewer->font->size.getValue()/scale);
  if(video) {
    QString str("Encoding video file to %1, please wait!");
    str=str.arg(QFileInfo(fileName).absoluteFilePath());
    statusBar()->showMessage(str);
    msg(Info)<<str.toStdString()<<endl;
    QString videoCmd=dialog.getVideoCmd();
    QFile(QFileInfo(fileName).absoluteFilePath()).remove();
    videoCmd.replace("%O", QFileInfo(fileName).absoluteFilePath());
    videoCmd.replace("%B", QString::number(dialog.getBitRate()*1024));
    videoCmd.replace("%F", QString::number(fps, 'f', 1));
    msg(Info)<<"Running command:"<<endl
             <<videoCmd.toStdString()<<endl
             <<"in directory "<<tmpDir->path().toStdString()<<endl;
    QString cwd=QDir::currentPath();
    QDir::setCurrent(tmpDir->path());
    int ret=std::system(videoCmd.toStdString().c_str());
    QDir::setCurrent(cwd);
    if(ret!=0) {
      QString str("FAILED. See console output!");
      statusBar()->showMessage(str, 10000);
      msg(Info)<<str.toStdString()<<endl;
      return;
    }
  }
}

void MainWindow::stopSCSlot() {
  if(hdf5RefreshDelta>0)
    hdf5RefreshTimer->start(hdf5RefreshDelta);
  animTimer->stop();
  stopAct->setChecked(true);
  lastFrameAct->setChecked(false);
  playAct->setChecked(false);
}

void MainWindow::lastFrameSCSlot() {
  hdf5RefreshTimer->stop();
  if(!lastFrameAct->isChecked()) {
    stopAct->setChecked(true);
    animTimer->stop();
    return;
  }

  stopAct->setChecked(false);
  playAct->setChecked(false);
  if(fpsMax<1e-15)
    animTimer->start();
  else
    animTimer->start((int)(1000/fpsMax));
}

void MainWindow::playSCSlot() {
  if(hdf5RefreshDelta>0)
    hdf5RefreshTimer->start(hdf5RefreshDelta);
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
  speedSB->setValue(speedSB->value()*appSettings->get<double>(AppSettings::speedChangeFactor));
}

void MainWindow::speedDownSlot() {
  speedSB->setValue(speedSB->value()/appSettings->get<double>(AppSettings::speedChangeFactor));
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
    setCameraType(SoOrthographicCamera::getClassTypeId());
    appSettings->set(AppSettings::cameraType, 0);
    glViewer->changeCameraValues((SoCamera*)newCamera);
  }
  else if(newCamera->getTypeId()==SoPerspectiveCamera::getClassTypeId()) {
    setCameraType(SoPerspectiveCamera::getClassTypeId());
    appSettings->set(AppSettings::cameraType, 1);
    glViewer->changeCameraValues((SoCamera*)newCamera);
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
    auto acts=((Object*)obj->child(i))->getProperties()->getActions();
    for(auto &act : acts) {
      if(act->objectName()=="Body::outLine")
        act->setChecked(enableOutLine);
      if(act->objectName()=="Body::shilouetteEdge")
        act->setChecked(enableShilouetteEdge);
    }
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
  }
  else {
    *bgColor=*engDrawingBGColorSaved; // restore bg color
    *fgColorBottom=*engDrawingFGColorBottomSaved;
    *fgColorTop=*engDrawingFGColorTopSaved;
    engDrawing->whichChild.setValue(SO_SWITCH_NONE); // disable engineering drawing
    setOutLineAndShilouetteEdgeRecursive(objectList->invisibleRootItem(), true, false); // enable outline and disable shilouetteEdge
  }
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
  if(isSignalConnected(QMetaMethod::fromSignal(&MainWindow::objectSelected)))
    return;
  for(QTreeWidgetItemIterator it(objectList); *it; ++it) {
    auto obj=dynamic_cast<Object*>(*it);
    if(obj)
      obj->setHighlight((*it)->isSelected());
  }
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
      if (Fout.exists())
        openFile(Fout.fileName().toStdString());
    }
  }
  viewAllSlot();
}

void MainWindow::closeEvent(QCloseEvent *event) {
  appSettings->set(AppSettings::mainwindow_geometry, saveGeometry());
  if(!skipWindowState)
    appSettings->set(AppSettings::mainwindow_state, saveState());
  if(dialogStereo)
    dialogStereo->close();
  QMainWindow::closeEvent(event);
}

void MainWindow::showEvent(QShowEvent *event) {
  restoreGeometry(appSettings->get<QByteArray>(AppSettings::mainwindow_geometry));
  if(!skipWindowState)
    restoreState(appSettings->get<QByteArray>(AppSettings::mainwindow_state));
  QMainWindow::showEvent(event);
}

void MainWindow::shortAni() {
  int cur=shortAniElapsed->elapsed();
  if(cur==shortAniLast)
    return;
  shortAniLast=cur;
  double c=static_cast<double>(cur)/appSettings->get<int>(AppSettings::shortAniTime);
  if(c>=1) {
    shortAniTimer->stop();
    c=1;
  }
  if(shortAniFunc)
    shortAniFunc(c);
}

void MainWindow::startShortAni(const std::function<void(double)> &func, bool noAni) {
  if(appSettings->get<int>(AppSettings::shortAniTime)==0 || noAni) {
    func(1);
    return;
  }
  shortAniFunc=func;
  shortAniElapsed->start();
  shortAniTimer->start();
}

void MainWindow::setCameraType(SoType type) {
  glViewer->setCameraType(type);
  appSettings->set(AppSettings::cameraType, type == SoOrthographicCamera::getClassTypeId() ? 0 : 1);

  // 3D cursor scale
  if(glViewer->getCamera()->getTypeId()==SoOrthographicCamera::getClassTypeId()) {
    cursorScaleE->a.connectFrom(&static_cast<SoOrthographicCamera*>(glViewer->getCamera())->height);
    cursorScaleE->b.disconnect();
    cursorScaleE->c.connectFrom(&mouseCursorSizeField);
    cursorScaleE->expression.setValue("oA=vec3f(a, a, a)*c/100"); // oA=cursorScale->scaleFactor
  }
  else {
    cursorScaleE->a.connectFrom(&static_cast<SoPerspectiveCamera*>(glViewer->getCamera())->heightAngle);
    cursorScaleE->b.connectFrom(&static_cast<SoPerspectiveCamera*>(glViewer->getCamera())->focalDistance);
    cursorScaleE->c.connectFrom(&mouseCursorSizeField);
    cursorScaleE->expression.setValue("oA=vec3f(a, a, a)*b*c/100"); // oA=cursorScale->scaleFactor
  }
}

void MainWindow::setCursorPos(const SbVec3f *pos) {
  cursorSwitch->whichChild.setValue(pos ? SO_SWITCH_ALL : SO_SWITCH_NONE);
  if(pos)
    cursorPos->translation.setValue(*pos);
}

}
