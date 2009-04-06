#include "mainwindow.h"
#include <Inventor/Qt/SoQt.h>
#include <QtGui/QDockWidget>
#include <QtGui/QMenuBar>
#include <QtGui/QGridLayout>
#include <QtGui/QFileDialog>
#include <QtGui/QMouseEvent>
#include <QtGui/QApplication>
#include <Inventor/Qt/viewers/SoQtExaminerViewer.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoCone.h>
#include "object.h"
#include "cuboid.h"
#include "group.h"
#include "tinyxml.h"
#include "tinynamespace.h"
#include "objectfactory.h"
#include <string>

using namespace std;

MainWindow::MainWindow(int argc, char* argv[]) {
  // init SoQt and Inventor
  SoQt::init(this);

  // menu bar
  QMenuBar *menuBar=new QMenuBar(this);
  setMenuBar(menuBar);

  // file menu
  QMenu *fileMenu=new QMenu("File", menuBar);
  fileMenu->addAction("Add File...", this, SLOT(openFileDialog()));
  menuBar->addMenu(fileMenu);

  // gl viewer main widget
  QWidget *glViewerWG=new QWidget(this);
  SoQtExaminerViewer *glViewer=new SoQtExaminerViewer(glViewerWG);
  setCentralWidget(glViewerWG);
  sceneRoot=new SoSeparator;
  sceneRoot->ref();
  glViewer->setSceneGraph(sceneRoot);

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

  // read XML files
  for(int i=1; i<argc; i++)
    openFile(argv[i]);
}

void MainWindow::openFile(string fileName) {
  TiXmlDocument doc;
  doc.LoadFile(fileName);
  incorporateNamespace(doc.FirstChildElement());
  Object *object=ObjectFactory(doc.FirstChildElement());
  object->setText(0, fileName.c_str());
  objectList->addTopLevelItem(object);
  sceneRoot->addChild(object->getSoSwitch());
  objectList->expandAll();
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
