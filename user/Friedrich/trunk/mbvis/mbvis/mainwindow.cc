#include "config.h"
#include "mainwindow.h"
#include <Inventor/Qt/SoQt.h>
#include <QtGui/QDockWidget>
#include <QtGui/QMenuBar>
#include <QtGui/QGridLayout>
#include <QtGui/QFileDialog>
#include <QtGui/QMouseEvent>
#include <QtGui/QApplication>
#include <QtGui/QMessageBox>
#include <QtGui/QToolBar>
#include <QtGui/QLabel>
#include <QtGui/QDoubleSpinBox>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoCone.h>
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/nodes/SoEventCallback.h>
#include <Inventor/events/SoMouseButtonEvent.h>
#include <Inventor/events/SoLocation2Event.h>
#include <Inventor/sensors/SoFieldSensor.h>
#include "object.h"
#include "cuboid.h"
#include "group.h"
#include "tinyxml.h"
#include "tinynamespace.h"
#include "objectfactory.h"
#include <string>
#include <set>
#include <hdf5serie/fileserie.h>
#include <Inventor/SbViewportRegion.h>
#include <Inventor/actions/SoRayPickAction.h>
#include <Inventor/SoPickedPoint.h>

using namespace std;

MainWindow::Mode MainWindow::mode;
SoSeparator *MainWindow::sceneRootBBox;
QSlider *MainWindow::timeSlider;
double MainWindow::deltaTime;

MainWindow::MainWindow(int argc, char* argv[]) : QMainWindow() {
  setWindowTitle("MBVis - Multi Body Visualisation");

  // init SoQt and Inventor
  SoQt::init(this);

  // initialize global frame field
  Body::frame=(SoSFUInt32*)SoDB::createGlobalField("frame",SoSFUInt32::getClassTypeId());
  Body::frame->setValue(0);

  // main widget
  QWidget *mainWG=new QWidget(this);
  setCentralWidget(mainWG);
  QGridLayout *mainLO=new QGridLayout();
  mainWG->setLayout(mainLO);
  // gl viewer
  QWidget *glViewerWG=new QWidget(this);
  glViewer=new SoQtMyViewer(glViewerWG);
  mainLO->addWidget(glViewerWG,0,0);
  sceneRoot=new SoSeparator;
  sceneRoot->ref();
  sceneRootBBox=new SoSeparator;
  sceneRoot->addChild(sceneRootBBox);
  SoBaseColor *color=new SoBaseColor;
  color->rgb.setValue(0,1,0);
  sceneRootBBox->addChild(color);
  SoDrawStyle *style=new SoDrawStyle;
  style->style.setValue(SoDrawStyle::LINES);
  sceneRootBBox->addChild(style);
  glViewer->setSceneGraph(sceneRoot);
  // time slider
  timeSlider=new QSlider(Qt::Vertical, this);
  mainLO->addWidget(timeSlider, 0, 1);
  timeSlider->setMinimum(0);
  connect(timeSlider, SIGNAL(valueChanged(int)), this, SLOT(updateFrame(int)));

  // object list dock widget
  QDockWidget *objectListDW=new QDockWidget(tr("Objects"),this);
  QWidget *objectListWG=new QWidget;
  QGridLayout *objectListLO=new QGridLayout;
  objectListWG->setLayout(objectListLO);
  objectListDW->setWidget(objectListWG);
  addDockWidget(Qt::LeftDockWidgetArea,objectListDW);
  objectList = new QTreeWidget(objectListDW);
  objectListLO->addWidget(objectList, 0,0);
  objectList->setHeaderHidden(true);
  connect(objectList,SIGNAL(pressed(QModelIndex)), this, SLOT(objectListClicked()));

  // object info dock widget
  QDockWidget *objectInfoDW=new QDockWidget(tr("Object Info"),this);
  QWidget *objectInfoWG=new QWidget;
  QGridLayout *objectInfoLO=new QGridLayout;
  objectInfoWG->setLayout(objectInfoLO);
  objectInfoDW->setWidget(objectInfoWG);
  addDockWidget(Qt::LeftDockWidgetArea,objectInfoDW);
  objectInfo = new QTextEdit(objectInfoDW);
  objectInfoLO->addWidget(objectInfo, 0,0);
  objectInfo->setReadOnly(true);
  objectInfo->setLineWrapMode(QTextEdit::NoWrap);
  connect(objectList,SIGNAL(currentItemChanged(QTreeWidgetItem*,QTreeWidgetItem*)),this,SLOT(setObjectInfo(QTreeWidgetItem*)));

  // menu bar
  QMenuBar *menuBar=new QMenuBar(this);
  setMenuBar(menuBar);

  // file menu
  QMenu *fileMenu=new QMenu("File", menuBar);
  QAction *addFileAct=fileMenu->addAction(QIcon(":/addfile.svg"), "Add File...", this, SLOT(openFileDialog()));
  fileMenu->addSeparator();
  fileMenu->addAction(QIcon(":/quit.svg"), "Exit", qApp, SLOT(quit()));
  menuBar->addMenu(fileMenu);

  // animation menu
  animGroup=new QActionGroup(this);
  QAction *stopAct=new QAction(QIcon(":/stop.svg"), "Stop", animGroup);
  QAction *lastFrameAct=new QAction(QIcon(":/lastframe.svg"), "Last Frame", animGroup);
  QAction *playAct=new QAction(QIcon(":/play.svg"), "Play", animGroup);
  stopAct->setCheckable(true);
  stopAct->setData(QVariant(stop));
  playAct->setCheckable(true);
  playAct->setData(QVariant(play));
  lastFrameAct->setCheckable(true);
  lastFrameAct->setData(QVariant(lastFrame));
  stopAct->setChecked(true);
  QMenu *animationMenu=new QMenu("Animation", menuBar);
  animationMenu->addAction(stopAct);
  animationMenu->addAction(lastFrameAct);
  animationMenu->addAction(playAct);
  menuBar->addMenu(animationMenu);
  connect(animGroup,SIGNAL(triggered(QAction*)),this,SLOT(animationButtonSlot(QAction*)));

  // view menu
  QMenu *viewMenu=new QMenu("View", menuBar);
  QAction *viewAllAct=viewMenu->addAction(QIcon(":/viewall.svg"),"View All", this, SLOT(viewAllSlot()));
  viewMenu->addSeparator()->setText("Parallel View");
  QAction *topViewAct=viewMenu->addAction(QIcon(":/topview.svg"),"Top-View", this, SLOT(viewTopSlot()));
  QAction *bottomViewAct=viewMenu->addAction(QIcon(":/bottomview.svg"),"Bottom-View", this, SLOT(viewBottomSlot()));
  QAction *frontViewAct=viewMenu->addAction(QIcon(":/frontview.svg"),"Front-View", this, SLOT(viewFrontSlot()));
  QAction *backViewAct=viewMenu->addAction(QIcon(":/backview.svg"),"Back-View", this, SLOT(viewBackSlot()));
  QAction *rightViewAct=viewMenu->addAction(QIcon(":/rightview.svg"),"Right-View", this, SLOT(viewRightSlot()));
  QAction *leftViewAct=viewMenu->addAction(QIcon(":/leftview.svg"),"Left-View", this, SLOT(viewLeftSlot()));
  viewMenu->addSeparator();
  QAction *cameraAct=viewMenu->addAction(QIcon(":/camera.svg"),"Toggle Camera Type", this, SLOT(toggleCameraTypeSlot()));
  menuBar->addMenu(viewMenu);

  // dock menu
  QMenu *dockMenu=new QMenu("Docks", menuBar);
  dockMenu->addAction(objectListDW->toggleViewAction());
  dockMenu->addAction(objectInfoDW->toggleViewAction());
  menuBar->addMenu(dockMenu);

  // file toolbar
  QToolBar *fileTB=new QToolBar("FileToolBar", this);
  addToolBar(Qt::TopToolBarArea, fileTB);
  fileTB->addAction(addFileAct);

  // view toolbar
  QToolBar *viewTB=new QToolBar("ViewToolBar", this);
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
  viewTB->addAction(cameraAct);

  // animation toolbar
  QToolBar *animationTB=new QToolBar("AnimationToolBar", this);
  addToolBar(Qt::TopToolBarArea, animationTB);
  // stop button
  animationTB->addAction(stopAct);
  // last frame button
  animationTB->addSeparator();
  animationTB->addAction(lastFrameAct);
  // play button
  animationTB->addSeparator();
  animationTB->addAction(playAct);
  // speed spin box
  speedSB=new QDoubleSpinBox;
  speedSB->setRange(1e-30, 1e30);
  speedSB->setMaximumSize(50, 1000);
  speedSB->setDecimals(3);
  speedSB->setButtonSymbols(QDoubleSpinBox::NoButtons);
  speedSB->setValue(1.0);
  connect(speedSB, SIGNAL(valueChanged(double)), this, SLOT(speedChanged(double)));
  QWidget *speedWG=new QWidget(this);
  QGridLayout *speedLO=new QGridLayout(speedWG);
  speedLO->setSpacing(0);
  speedLO->setContentsMargins(0,0,0,0);
  speedWG->setLayout(speedLO);
  QLabel *speedL=new QLabel("Speed:", this);
  speedLO->addWidget(speedL, 0, 0);
  speedLO->addWidget(speedSB, 1, 0);
#ifdef HAVE_QWT5_QWT_WHEEL_H
  speedWheel=new QwtWheel(this);
  speedWheel->setWheelWidth(10);
  speedWheel->setOrientation(Qt::Vertical);
  speedWheel->setRange(-2, 2, 0.0001);
  speedWheel->setTotalAngle(360*15);
  connect(speedWheel, SIGNAL(valueChanged(double)), this, SLOT(speedWheelChanged(double)));
  connect(speedWheel, SIGNAL(sliderPressed()), this, SLOT(speedWheelPressed()));
  connect(speedWheel, SIGNAL(sliderReleased()), this, SLOT(speedWheelReleased()));
  speedLO->addWidget(speedWheel, 0, 1, 2, 1);
#endif
  animationTB->addWidget(speedWG);
  // frame spin box
  animationTB->addSeparator();
  frameSB=new QSpinBox;
  frameSB->setMinimumSize(55,0);
  QWidget *frameWG=new QWidget(this);
  QGridLayout *frameLO=new QGridLayout(frameWG);
  frameLO->setSpacing(0);
  frameLO->setContentsMargins(0,0,0,0);
  frameWG->setLayout(frameLO);
  QLabel *frameL=new QLabel("Frame:", this);
  frameLO->addWidget(frameL, 0, 0);
  frameLO->addWidget(frameSB, 1, 0);
  animationTB->addWidget(frameWG);
  connect(frameSB, SIGNAL(valueChanged(int)), this, SLOT(updateFrame(int)));
  connect(timeSlider, SIGNAL(valueChanged(int)), frameSB, SLOT(setValue(int)));
  connect(timeSlider, SIGNAL(rangeChanged(int,int)), this, SLOT(frameSBSetRange(int,int)));

  // tool menu
  QMenu *toolMenu=new QMenu("Tools", menuBar);
  toolMenu->addAction(fileTB->toggleViewAction());
  toolMenu->addAction(viewTB->toggleViewAction());
  toolMenu->addAction(animationTB->toggleViewAction());
  menuBar->addMenu(toolMenu);

  // help menu
  menuBar->addSeparator();
  QMenu *helpMenu=new QMenu("Help", menuBar);
  helpMenu->addAction("About MBVis...", this, SLOT(aboutMBVis()));
  menuBar->addMenu(helpMenu);

  // register callback function on frame change
  SoFieldSensor *sensor=new SoFieldSensor(frameSensorCB, this);
  sensor->attach(Body::frame);

  // animation timer
  animTimer=new QTimer(this);
  connect(animTimer, SIGNAL(timeout()), this, SLOT(heavyWorkSlot()));
  time=new QTime;

  // read XML files
  for(int i=1; i<argc; i++)
    openFile(argv[i]);

  glViewer->viewAll();
  resize(640,480);
}

void MainWindow::openFile(string fileName) {
  // open HDF5
  H5::FileSerie *h5File=new H5::FileSerie(fileName.substr(0,fileName.length()-string(".amvis.xml").length())+".amvis.h5", H5F_ACC_RDONLY);
  H5::Group *h5Parent=(H5::Group*)h5File;
  // read XML
  TiXmlDocument doc;
  doc.LoadFile(fileName);
  incorporateNamespace(doc.FirstChildElement());
  Object *object=ObjectFactory(doc.FirstChildElement(), h5Parent);
  object->setText(0, fileName.c_str());
  object->iconFile=":/h5file.svg";
  object->setIcon(0, QIcon(object->iconFile.c_str()));
  objectList->addTopLevelItem(object);
  sceneRoot->addChild(object->getSoSwitch());
  objectList->expandAll();

  // force a update
  Body::frame->touch();
}

void MainWindow::openFileDialog() {
  QStringList files=QFileDialog::getOpenFileNames(0, "Open AMVis Files", ".", "AMVis Files (*.amvis.xml)");
  for(int i=0; i<files.size(); i++)
    openFile(files[i].toStdString());
}

void MainWindow::objectListClicked() {
  if(QApplication::mouseButtons()==Qt::RightButton) {
    Object *object=(Object*)objectList->currentItem();
    QMenu* menu=object->createMenu();
    menu->exec(QCursor::pos());
    delete menu;
  }
}

void MainWindow::aboutMBVis() {
  QMessageBox::about(this, "About MBVis",
    "<h1>MBVis - Multi Body Visualisation</h1>"
    "<p>Copyright &copy; Markus Friedrich <tt>&lt;mafriedrich@user.berlios.de&gt;</tt><p/>"
    "<p>Licensed under the General Public License (see file COPYING).</p>"
    "<p>This is free software; see the source for copying conditions.  There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.</p>"
    "<h2>Authors:</h2>"
    "<ul>"
    "  <li>Markus Friedrich <tt>&lt;mafriedrich@users.berlios.de&gt;</tt> (Maintainer)</li>"
    "</ul>"
    "<h2>This program uses:</h2>"
    "<ul>"
    "  <li>'Qt - A cross-platform application and UI framework' by Nokia from <tt>http://www.qtsoftware.com</tt> (License: GPL/LGPL)</li>"
    "  <li>'Coin - An OpenGL based, retained mode 3D graphics rendering library' by Kongsberg SIM from <tt>http://www.coin3d.org</tt> (License: GPL)</li>"
    "  <li>'SoQt - A Qt GUI component toolkit library for Coin' by Kongsberg SIM from <tt>http://www.coin3d.org</tt> (License: GPL)</li>"
    "  <li>'HDF5Serie - A HDF5 Wrapper for Time Series' by Markus Friedrich from <tt>http://hdf5serie.berlios.de</tt> (License: LGPL)</li>"
    "  <li>'HDF - Hierarchical Data Format' by The HDF Group from <tt>http://www.hdfgroup.org</tt> (License: NCSA-HDF)</li>"
    "  <li>'TinyXML - A simple, small, C++ XML parser' by Lee Thomason from <tt>http://www.grinninglizard.com/tinyxml</tt> (Licence: ZLib)</li>"
    "  <li>...</li>"
    "</ul>"
    "<p>A special thanks to all authors of this projects.</p>"
  );
}

void MainWindow::viewParallel(ViewSide side) {
  switch(side) {
    case top:    glViewer->getCamera()->position.setValue(0,0,+1); break;
    case bottom: glViewer->getCamera()->position.setValue(0,0,-1); break;
    case front:  glViewer->getCamera()->position.setValue(0,-1,0); break;
    case back:   glViewer->getCamera()->position.setValue(0,+1,0); break;
    case right:  glViewer->getCamera()->position.setValue(+1,0,0); break;
    case left:   glViewer->getCamera()->position.setValue(-1,0,0); break;
  }
  glViewer->getCamera()->pointAt(SbVec3f(0,0,0));
  glViewer->viewAll();
}

bool MainWindow::soQtEventCB(const SoEvent *const event) {
  // if mouse button event
  if(event->isOfType(SoMouseButtonEvent::getClassTypeId())) {
    SoMouseButtonEvent *ev=(SoMouseButtonEvent*)event;
    // if Ctrl|Ctrl+Alt + Button1|Button2 + Pressed: select object (show list if Alt)
    if(ev->wasCtrlDown() && ev->getState()==SoButtonEvent::DOWN &&
       (ev->getButton()==SoMouseButtonEvent::BUTTON1 ||
        ev->getButton()==SoMouseButtonEvent::BUTTON2)) {
      // get picked points by ray
      SoRayPickAction *pickAction=new SoRayPickAction(glViewer->getViewportRegion());
      pickAction->setPoint(ev->getPosition());
      pickAction->setRadius(3.0);
      pickAction->setPickAll(true);
      pickAction->apply(glViewer->getSceneManager()->getSceneGraph());
      SoPickedPointList pickedPoints=pickAction->getPickedPointList();
      // get objects by point/path
      set<Object*> pickedObject;
      float x=1e99, y=1e99, z=1e99, xOld, yOld, zOld;
      cout<<"Clicked points:"<<endl;
      for(int i=0; pickedPoints[i]; i++) {
        SoPath *path=pickedPoints[i]->getPath();
        bool found=false;
        for(int j=path->getLength()-1; j>=0; j--) {
          map<SoNode*,Object*>::iterator it=Object::objectMap.find(path->getNode(j));
          if(it!=Object::objectMap.end()) {
            pickedObject.insert(it->second);
            found=true;
            break;
          }
        }
        if(!found) continue;
        xOld=x; yOld=y; zOld=z;
        pickedPoints[i]->getPoint().getValue(x,y,z);
        if(fabs(x-xOld)>1e-7 || fabs(y-yOld)>1e-7 || fabs(z-zOld)>1e-7)
          cout<<"Point on: "<<(*(--pickedObject.end()))->getPath()<<": "<<x<<" "<<y<<" "<<z<<endl;
        if(!ev->wasAltDown()) break;
      }
      if(pickedObject.size()>0) {
        // if Button2 show menu of picked objects
        if(ev->wasAltDown()) {
          QMenu *menu=new QMenu(this);
          int ind=0;
          set<Object*>::iterator it;
          for(it=pickedObject.begin(); it!=pickedObject.end(); it++) {
            QAction *action=new QAction((*it)->icon(0),(*it)->getPath().c_str(),menu);
            action->setData(QVariant(ind++));
            menu->addAction(action);
          }
          QAction *action=menu->exec(QCursor::pos());
          if(action==0) return true;
          ind=action->data().toInt();
          it=pickedObject.begin();
          for(int i=0; i<ind; i++, it++);
          objectList->setCurrentItem(*it);
          delete menu;
        }
        // if Button1 select picked object
        else
          objectList->setCurrentItem(*pickedObject.begin());
        // if Button2 show property menu
        if(ev->getButton()==SoMouseButtonEvent::BUTTON2) {
          Object *object=(Object*)(objectList->currentItem());
          QMenu* menu=object->createMenu();
          menu->exec(QCursor::pos());
          delete menu;
        }
      }
      return true; // event applied
    }
    // if released: set mode=no
    if(ev->getState()==SoButtonEvent::UP) {
      if(mode==no); // release from no
      if(mode==zoom) { // release from zoom
        ev->setButton(SoMouseButtonEvent::BUTTON3);
        ev->setCtrlDown(true);
      }
      if(mode==rotate) // release from rotate
        ev->setButton(SoMouseButtonEvent::BUTTON1);
      if(mode==translate) // release from translate
        ev->setButton(SoMouseButtonEvent::BUTTON3);
      mode=no;
      return false; // pass event to base class
    }
    // if Button3 pressed: enter zoom
    if(ev->getButton()==SoMouseButtonEvent::BUTTON3 && ev->getState()==SoButtonEvent::DOWN) {
      ev->setButton(SoMouseButtonEvent::BUTTON3);
      ev->setCtrlDown(true);
      mode=zoom;
      return false; // pass event to base class
    }
    // if Button1 pressed: enter rotate
    if(ev->getButton()==SoMouseButtonEvent::BUTTON1 && ev->getState()==SoButtonEvent::DOWN) {
      ev->setButton(SoMouseButtonEvent::BUTTON1);
      mode=rotate;
      return false; // pass event to base class
    }
    // if Button2 pressed: enter tranlate
    if(ev->getButton()==SoMouseButtonEvent::BUTTON2 && ev->getState()==SoButtonEvent::DOWN) {
      ev->setButton(SoMouseButtonEvent::BUTTON3);
      mode=translate;
      return false; // pass event to base class
    }
  }
  // if mouse move event
  if(event->isOfType(SoLocation2Event::getClassTypeId())) {
    SoLocation2Event *ev=(SoLocation2Event*)event;
    if(mode==zoom)
      ev->setCtrlDown(true);
    return false; // pass event to base class
  }
  // if keyboard event
  if(event->isOfType(SoKeyboardEvent::getClassTypeId())) {
    return true; // no nothing
  }
  return false; // pass event to base class
}

void MainWindow::frameSensorCB(void *data, SoSensor*) {
  MainWindow *me=(MainWindow*)data;
  me->setObjectInfo(me->objectList->currentItem());
  me->timeSlider->setValue(Body::frame->getValue());
}

void MainWindow::animationButtonSlot(QAction* act) {
  if(act->data().toInt()==stop)
    animTimer->stop();
  else if(act->data().toInt()==lastFrame) {
    printf("Not implemented yet\n");
    animTimer->stop();
  }
  else {
    animStartFrame=Body::frame->getValue();
    time->restart();
    animTimer->start();
  }
}

void MainWindow::speedChanged(double value) {
  if(animGroup->checkedAction()->data().toInt()==play) {
    // emulate anim stop click
    animTimer->stop();
    // emulate anim play click
    animStartFrame=Body::frame->getValue();
    time->restart();
    animTimer->start();
  }
}

void MainWindow::heavyWorkSlot() {
  if(animGroup->checkedAction()->data().toInt()==play) {
    double dT=time->elapsed()/1000.0*speedSB->value();// time since play click
    int dframe=(int)(dT/deltaTime);// frame increment since play click
    int frame=(animStartFrame+dframe)%(timeSlider->maximum()+1); // frame number
    Body::frame->setValue(frame); // set frame => update scene
  }
}

void MainWindow::speedWheelChanged(double value) {
#ifdef HAVE_QWT5_QWT_WHEEL_H
  printf("XXX %f\n", pow(10,value));
  speedSB->setValue(oldSpeed*pow(10,value));
#endif
}

void MainWindow::speedWheelPressed() {
#ifdef HAVE_QWT5_QWT_WHEEL_H
  oldSpeed=speedSB->value();
#endif
}

void MainWindow::speedWheelReleased() {
#ifdef HAVE_QWT5_QWT_WHEEL_H
  oldSpeed=speedSB->value();
  speedWheel->setValue(0);
#endif
}
