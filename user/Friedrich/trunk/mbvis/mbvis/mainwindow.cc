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
#include "object.h"
#include "cuboid.h"
#include "group.h"
#include "tinyxml.h"
#include "tinynamespace.h"
#include "objectfactory.h"
#include <string>
#include <hdf5serie/fileserie.h>
  //////////////
#include <Inventor/SbViewportRegion.h>
#include <Inventor/actions/SoRayPickAction.h>
#include <Inventor/SoPickedPoint.h>
  //////////////

using namespace std;

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
  viewMenu->addSeparator();
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
  menuBar->addMenu(dockMenu);

  // help menu
  menuBar->addSeparator();
  QMenu *helpMenu=new QMenu("Help", menuBar);
  helpMenu->addAction("About MBVis...", this, SLOT(aboutMBVis()));
  menuBar->addMenu(helpMenu);

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
  object->setIcon(0, QIcon(":/h5file.svg"));
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
  //////////////
  SbViewportRegion vpr=glViewer->getViewportRegion();
  SoRayPickAction *rpa=new SoRayPickAction(vpr);
  rpa->setPoint(SbVec2s(0,0));
  rpa->setRadius(3.0);
  rpa->apply(glViewer->getSceneManager()->getSceneGraph());
  SoPickedPoint *pp=rpa->getPickedPoint();
  if(pp) {
    float x, y, z;
    pp->getPoint().getValue(x,y,z);
    printf("XXX %f %f %f\n", x, y, z);
    SoPath *p=pp->getPath();
    for(int i=p->getLength()-1; i>=0; i--) {
      map<SoNode*,Object*>::iterator it=Object::objectMap.find(p->getNode(i));
      if(it!=Object::objectMap.end()) {
        it->second->setSelected(true);//use objectList->setCurrentItem(...)
        break;
      }
    }
  }
  //////////////
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

void MainWindow::viewParallel(int i) {
  switch(i) {
    case 0: glViewer->getCamera()->position.setValue(0,0,1); break;
    case 1: glViewer->getCamera()->position.setValue(0,0,-1); break;
    case 2: glViewer->getCamera()->position.setValue(0,-1,0); break;
    case 3: glViewer->getCamera()->position.setValue(0,1,0); break;
    case 4: glViewer->getCamera()->position.setValue(1,0,0); break;
    case 5: glViewer->getCamera()->position.setValue(-1,0,0); break;
  }
  glViewer->getCamera()->pointAt(SbVec3f(0,0,0));
  glViewer->viewAll();
}
