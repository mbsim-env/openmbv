#include "mainwindow.h"
#include <Inventor/Qt/SoQt.h>
#include <QtGui/QDockWidget>
#include <QtGui/QMenuBar>
#include <QtGui/QGridLayout>
#include <QtGui/QFileDialog>
#include <QtGui/QMouseEvent>
#include <QtGui/QApplication>
#include <QtGui/QMessageBox>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoCone.h>
#include <Inventor/nodes/SoCamera.h>
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
  //glViewer=new SoQtExaminerViewer(glViewerWG);
  glViewer=new SoQtMyViewer(glViewerWG);
  mainLO->addWidget(glViewerWG,0,0);
  sceneRoot=new SoSeparator;
  sceneRoot->ref();
  glViewer->setSceneGraph(sceneRoot);
  // time slider
  QSlider *timeSlider=new QSlider(Qt::Vertical, this);
  mainLO->addWidget(timeSlider, 0, 1);
  timeSlider->setMinimum(0);
  timeSlider->setMaximum(4000); //TODO set by max hdf5
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
  fileMenu->addAction("Add File...", this, SLOT(openFileDialog()));
  fileMenu->addSeparator();
  fileMenu->addAction("Exit", qApp, SLOT(quit()));
  menuBar->addMenu(fileMenu);

  // view menu
  QMenu *viewMenu=new QMenu("View", menuBar);
  viewMenu->addAction("View All", this, SLOT(viewAllSlot()));
  viewMenu->addSeparator()->setText("Parallel View");
  viewMenu->addAction("Top-View", this, SLOT(viewTopSlot()));
  viewMenu->addAction("Bottom-View", this, SLOT(viewBottomSlot()));
  viewMenu->addAction("Front-View", this, SLOT(viewFrontSlot()));
  viewMenu->addAction("Back-View", this, SLOT(viewBackSlot()));
  viewMenu->addAction("Right-View", this, SLOT(viewRightSlot()));
  viewMenu->addAction("Left-View", this, SLOT(viewLeftSlot()));
  viewMenu->addSeparator();
  viewMenu->addAction("Toggle Camera Type", this, SLOT(toggleCameraTypeSlot()));
  menuBar->addMenu(viewMenu);

  // dock menu
  QMenu *dockMenu=new QMenu("Docks", menuBar);
  dockMenu->addAction(objectListDW->toggleViewAction());
  dockMenu->addAction(objectInfoDW->toggleViewAction());
  menuBar->addMenu(dockMenu);

  // help menu
  menuBar->addSeparator();
  QMenu *helpMenu=new QMenu("Help", menuBar);
  helpMenu->addAction("About MBVis...", this, SLOT(aboutMBVis()));
  menuBar->addMenu(helpMenu);
  
  // register callback function on frame change
  SoFieldSensor *sensor=new SoFieldSensor(frameSensorCB, this);
  sensor->attach(Body::frame);

  // read XML files
  for(int i=1; i<argc; i++)
    openFile(argv[i]);

  glViewer->viewAll();
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
    "  <li>'AutoTools - Build System' by Free Software Foundation from <tt>http://www.gnu.org</tt> (Licence: GPL)</li>"
    "  <li>'AutoTroll - Build Qt apps with the autotools (Autoconf/Automake)' by Benoid Sigoure from <tt>http://www.tsunanet.net/autotroll</tt> (Licence: GPL)</li>"
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
    // if Ctrl + Button1|Button2 + Pressed: pick object
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
      for(int i=0; pickedPoints[i] && (i<1 || ev->getButton()==SoMouseButtonEvent::BUTTON2); i++) {
        SoPath *path=pickedPoints[i]->getPath();
        for(int j=path->getLength()-1; j>=0; j--) {
          map<SoNode*,Object*>::iterator it=Object::objectMap.find(path->getNode(j));
          if(it!=Object::objectMap.end()) {
            pickedObject.insert(it->second);
            break;
          }
        }
        xOld=x; yOld=y; zOld=z;
        pickedPoints[i]->getPoint().getValue(x,y,z);
        if(fabs(x-xOld)>1e-7 || fabs(y-yOld)>1e-7 || fabs(z-zOld)>1e-7)
          cout<<"Point on: "<<(*(--pickedObject.end()))->getPath()<<": "<<x<<" "<<y<<" "<<z<<endl;
      }
      if(pickedObject.size()>0) {
        // if Button2 show menu of picked objects
        if(ev->getButton()==SoMouseButtonEvent::BUTTON2) {
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
}
