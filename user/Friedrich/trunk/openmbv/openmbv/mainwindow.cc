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
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QStatusBar>
#include <QWebHistory>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoCone.h>
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/nodes/SoEventCallback.h>
#include <Inventor/nodes/SoLightModel.h>
#include <Inventor/events/SoMouseButtonEvent.h>
#include <Inventor/events/SoLocation2Event.h>
#include <Inventor/sensors/SoFieldSensor.h>
#include "exportdialog.h"
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

MainWindow *MainWindow::instance=0;

MainWindow::MainWindow(list<string>& arg) : QMainWindow(), mode(no), deltaTime(0), fpsMax(25), oldSpeed(1), helpViewer(0) {
  if(instance) { cout<<"The class MainWindow is a singleton class!"<<endl; _exit(1); }
  instance=this;

  setWindowTitle("OpenMBV - Open Multi Body Viewer");
  setWindowIcon(QIcon(":/openmbv.svg"));

  // init SoQt and Inventor
  SoQt::init(this);
  // init realtime
  SoDB::enableRealTimeSensor(false);
  SoSceneManager::enableRealTimeUpdate(false);

  // initialize global frame field
  frame=(SoSFUInt32*)SoDB::createGlobalField("frame",SoSFUInt32::getClassTypeId());
  frame->setValue(0);

  // main widget
  QWidget *mainWG=new QWidget(this);
  setCentralWidget(mainWG);
  QGridLayout *mainLO=new QGridLayout();
  mainWG->setLayout(mainLO);
  // gl viewer
  QWidget *glViewerWG=new QWidget(this);
  timeString=new SoText2;
  timeString->ref();
  glViewer=new SoQtMyViewer(glViewerWG, timeString);
  mainLO->addWidget(glViewerWG,0,0);
  sceneRoot=new SoSeparator;
  sceneRoot->ref();
  sceneRootBBox=new SoSeparator;
  sceneRoot->addChild(sceneRootBBox);
  SoLightModel *lm=new SoLightModel;
  sceneRootBBox->addChild(lm);
  lm->model.setValue(SoLightModel::BASE_COLOR);
  SoBaseColor *color=new SoBaseColor;
  sceneRootBBox->addChild(color);
  color->rgb.setValue(0,1,0);
  SoDrawStyle *style=new SoDrawStyle;
  style->style.setValue(SoDrawStyle::LINES);
  sceneRootBBox->addChild(style);
  glViewer->setSceneGraph(sceneRoot);
  
  // time slider
  timeSlider=new QSlider(Qt::Vertical, this);
  mainLO->addWidget(timeSlider, 0, 1);
  timeSlider->setMinimum(0);
  connect(timeSlider, SIGNAL(sliderMoved(int)), this, SLOT(updateFrame(int)));

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
  fileMenu->addAction(QIcon(":/exportimg.svg"), "Export PNG (current frame)...", this, SLOT(exportCurrentAsPNG()));
  fileMenu->addAction(QIcon(":/exportimgsequence.svg"), "Export PNG sequence...", this, SLOT(exportSequenceAsPNG()));
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
#ifdef HAVE_QWT_WHEEL_H
  speedWheel=new QwtWheel(this);
  speedWheel->setWheelWidth(10);
  speedWheel->setTotalAngle(360*15);
  connect(speedWheel, SIGNAL(valueChanged(double)), this, SLOT(speedWheelChangedD(double)));
#else
  speedWheel=new QSlider(this);
  speedWheel->setMaximumSize(15, 35);
  connect(speedWheel, SIGNAL(sliderMoved(int)), this, SLOT(speedWheelChanged(int)));
#endif
  speedWheel->setRange(-20000, 20000);
  speedWheel->setOrientation(Qt::Vertical);
  connect(speedWheel, SIGNAL(sliderPressed()), this, SLOT(speedWheelPressed()));
  connect(speedWheel, SIGNAL(sliderReleased()), this, SLOT(speedWheelReleased()));
  speedLO->addWidget(speedWheel, 0, 1, 2, 1);
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
  helpMenu->addAction(QIcon(":/help.svg"), "GUI Help...", this, SLOT(guiHelp()));
  helpMenu->addAction(QIcon(":/help.svg"), "XML Help...", this, SLOT(xmlHelp()));
  helpMenu->addAction(QIcon(":/openmbv.svg"), "About OpenMBV...", this, SLOT(aboutOpenMBV()));
  menuBar->addMenu(helpMenu);

  // status bar
  statusBar=new QStatusBar(this);
  fps=new QLabel("FPS: -");
  fpsTime=new QTime;
  statusBar->addPermanentWidget(fps);
  setStatusBar(statusBar);

  // register callback function on frame change
  SoFieldSensor *sensor=new SoFieldSensor(frameSensorCB, this);
  sensor->attach(frame);

  // animation timer
  animTimer=new QTimer(this);
  connect(animTimer, SIGNAL(timeout()), this, SLOT(heavyWorkSlot()));
  time=new QTime;

  // read XML files
  if(arg.empty()) arg.push_back("."); // if calles without argument loat current dir
  QDir dir;
  QRegExp filterRE("([^.]+\\.ombv.xml|[^.]+\\.ombv.env.xml)");
  dir.setFilter(QDir::Files);
  list<string>::iterator i=arg.begin(), i2;
  while(i!=arg.end()) {
    dir.setPath(i->c_str());
    if(dir.exists()) { // if directory
      // open all [^.]+\.ombv.xml or [^.]+\.ombv.env.xml files
      QStringList file=dir.entryList();
      for(int j=0; j<file.size(); j++)
        if(filterRE.exactMatch(file[j]))
          openFile(dir.path().toStdString()+"/"+file[j].toStdString());
      i2=i; i++; arg.erase(i2);
      continue;
    }
    if(QFile::exists(i->c_str())) {
      openFile(*i);
      i2=i; i++; arg.erase(i2);
      continue;
    }
    i++;
  }

  glViewer->viewAll();
  resize(640,480);
}

void MainWindow::openFile(string fileName) {
  // check file type
  bool env;
  if(fileName.length()>string(".ombv.xml").length() && fileName.substr(fileName.length()-string(".ombv.xml").length())==".ombv.xml")
    env=false;
  else if(fileName.length()>string(".ombv.env.xml").length() && fileName.substr(fileName.length()-string(".ombv.env.xml").length())==".ombv.env.xml")
    env=true;
  else {
    statusBar->showMessage(QString("Unknown file type: %1!").arg(fileName.c_str()), 2000);
    return;
  }

  H5::Group *h5Parent=0;
  if(!env) {
    // open HDF5
    H5::FileSerie *h5File=new H5::FileSerie(fileName.substr(0,fileName.length()-string(".ombv.xml").length())+".ombv.h5", H5F_ACC_RDONLY);
    h5Parent=(H5::Group*)h5File;
  }
  // read XML
  TiXmlDocument doc;
  doc.LoadFile(fileName);
  incorporateNamespace(doc.FirstChildElement());
  Object *object=ObjectFactory(doc.FirstChildElement(), h5Parent, objectList->invisibleRootItem(), sceneRoot);
  object->setText(0, fileName.c_str());
  object->getIconFile()=":/h5file.svg";
  object->setIcon(0, QIcon(object->getIconFile().c_str()));

  // force a update
  frame->touch();
}

void MainWindow::openFileDialog() {
  QStringList files=QFileDialog::getOpenFileNames(0, "OpenMBV Files", ".",
    "OpenMBV Files (*.ombv.xml *.ombv.env.xml);;"
    "OpenMBV Animation Files (*.ombv.xml);;"
    "OpenMBV Environment Files (*.ombv.env.xml)");
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

void MainWindow::guiHelp() {
  QMessageBox::information(this, "OpenMBV - GUI Help", 
    "<h1>GUI Help</h1>"
    "<h2>Mouse Interaction</h2>"
    "<ul>"
    "  <dt>Left-Button</dt><dd> Rotate the entiry scene</dd>"
    "  <dt>Right-Button</dt><dd> Translate the entiry scene</dd>"
    "  <dt>Middle-Button</dt><dd> Zoom the entiry scene</dd>"
    "  <dt>Ctrl+Left-Button</dt><dd> Selects the body under the cursor.</dd>"
    "  <dt>Crtl+Right-Button</dt><dd> Shows the property menu of body under the cursor.</dd>"
    "  <dt>Crtl+Alt+Left-Button</dt><dd> Shows a menu of all bodies under the cursor. Selecting one menu entry, selects this body.</dd>"
    "  <dt>Crtl+Alt+Right-Button</dt><dd> Shows a menu of all bodies under the cursor. Selecting one menu enety, shows the proptery menu of this body.</dd>"
    "  <dt>Crtl+Middle-Button</dt><dd> Seeks the focal point of the camera to the point on the shape under the cursor.</dd>"
    "</ul>");
}

void MainWindow::xmlHelp() {
  static QDialog *helpDialog=0;
  if(!helpDialog) {
    helpDialog=new QDialog(this);
    helpDialog->setWindowIcon(QIcon(":/help.svg"));
    helpDialog->setWindowTitle("OpenMBV - XML Help");
    QGridLayout *layout=new QGridLayout(helpDialog);
    helpDialog->setLayout(layout);
    QPushButton *home=new QPushButton("Home",helpDialog);
    layout->addWidget(home,0,0);
    QPushButton *helpBackward=new QPushButton("Backward",helpDialog);
    layout->addWidget(helpBackward,0,1);
    QPushButton *helpForward=new QPushButton("Forward",helpDialog);
    layout->addWidget(helpForward,0,2);
    helpViewer=new QWebView(helpDialog);
    layout->addWidget(helpViewer,1,0,1,3);
    connect(home, SIGNAL(clicked()), this, SLOT(helpHome()));
    connect(helpForward, SIGNAL(clicked()), helpViewer, SLOT(forward()));
    connect(helpBackward, SIGNAL(clicked()), helpViewer, SLOT(back()));
    helpViewer->load(QUrl("qrc:openmbv.html"));
  }
  helpDialog->show();
  helpDialog->raise();
  helpDialog->activateWindow();
  helpDialog->resize(700,500);
}

void MainWindow::helpHome() {
  helpViewer->load(QUrl("qrc:openmbv.html"));
}

void MainWindow::aboutOpenMBV() {
  QMessageBox::about(this, "About OpenMBV",
    "<h1>OpenMBV - Open Multi Body Viewer</h1>"
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
#ifdef HAVE_QWT_WHEEL_H
    "  <li>'Qwt - Qt Widgets for Technical Applications' by Uwe Rathmann from <tt>http://qwt.sourceforge.net</tt> (Licence: Qwt/LGPL)</li>"
#endif
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
        ev->getButton()==SoMouseButtonEvent::BUTTON2 ||
        ev->getButton()==SoMouseButtonEvent::BUTTON3)) {
      // get picked points by ray
      SoRayPickAction pickAction(glViewer->getViewportRegion());
      pickAction.setPoint(ev->getPosition());
      pickAction.setRadius(3.0);
      pickAction.setPickAll(true);
      pickAction.apply(glViewer->getSceneManager()->getSceneGraph());
      SoPickedPointList pickedPoints=pickAction.getPickedPointList();
      if(ev->getButton()==SoMouseButtonEvent::BUTTON3) { // seek to clicked point
        if(pickedPoints[0]==0) return true;
        glViewer->setSeekMode(true);
        glViewer->seekToPoint(pickedPoints[0]->getPoint());
        return true;
      }
      // get objects by point/path
      set<Object*> pickedObject;
      float x=1e99, y=1e99, z=1e99, xOld, yOld, zOld;
      cout<<"Clicked points:"<<endl;
      for(int i=0; pickedPoints[i]; i++) {
        SoPath *path=pickedPoints[i]->getPath();
        bool found=false;
        for(int j=path->getLength()-1; j>=0; j--) {
          map<SoNode*,Object*>::iterator it=Object::getObjectMap().find(path->getNode(j));
          if(it!=Object::getObjectMap().end()) {
            pickedObject.insert(it->second);
            found=true;
            break;
          }
        }
        if(!found) continue;
        xOld=x; yOld=y; zOld=z;
        pickedPoints[i]->getPoint().getValue(x,y,z);
        if(fabs(x-xOld)>1e-7 || fabs(y-yOld)>1e-7 || fabs(z-zOld)>1e-7) {
          cout<<"Point on: "<<(*(--pickedObject.end()))->getPath()<<": "<<x<<" "<<y<<" "<<z<<endl;
          statusBar->showMessage(QString("Point on: %1: %2,%3,%4").arg((*(--pickedObject.end()))->getPath().c_str()).arg(x).arg(y).arg(z), 2000);
        }
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
          delete menu;
          it=pickedObject.begin();
          for(int i=0; i<ind; i++, it++);
          objectList->setCurrentItem(*it);
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
    fps->setText(QString("FPS: >25(max)"));
  else
    fps->setText(QString("FPS: %1").arg(fpsOut, 0, 'f', 1));

  count=1;
}

void MainWindow::animationButtonSlot(QAction* act) {
  if(act->data().toInt()==stop) {
    animTimer->stop();
  }
  else if(act->data().toInt()==lastFrame) {
    printf("Not implemented yet!!!!!!!!!!!!!!\n"); //TODO
    animTimer->stop();
  }
  else {
    animStartFrame=frame->getValue();
    time->restart();
    if(fpsMax<1e-15)
      animTimer->start();
    else
      animTimer->start((int)(1000/fpsMax));
  }
}

void MainWindow::speedChanged(double value) {
  if(animGroup->checkedAction()->data().toInt()==play) {
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
  if(animGroup->checkedAction()->data().toInt()==play) {
    double dT=time->elapsed()/1000.0*speedSB->value();// time since play click
    int dframe=(int)(dT/deltaTime);// frame increment since play click
    int frame_=(animStartFrame+dframe)%(timeSlider->maximum()+1); // frame number
    frame->setValue(frame_); // set frame => update scene
    //glViewer->render(); // force rendering
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

void MainWindow::exportAsPNG(SoOffscreenRenderer &myRenderer, std::string fileName, bool transparent, float red, float green, float blue) {
  if(transparent)
    myRenderer.setComponents(SoOffscreenRenderer::RGB_TRANSPARENCY);
  else
    myRenderer.setBackgroundColor(SbColor(red,green,blue));
  myRenderer.render(glViewer->getSceneManager()->getSceneGraph());
  short width, height;
  myRenderer.getViewportRegion().getWindowSize().getValue(width, height);
  if(transparent) {
    for(int i=0; i<width*height*4; i+=4) {
      unsigned char r=myRenderer.getBuffer()[i+0];
      unsigned char g=myRenderer.getBuffer()[i+1];
      unsigned char b=myRenderer.getBuffer()[i+2];
      unsigned char a=myRenderer.getBuffer()[i+3];
      myRenderer.getBuffer()[i+0]=b;
      myRenderer.getBuffer()[i+1]=g;
      myRenderer.getBuffer()[i+2]=r;
      myRenderer.getBuffer()[i+3]=a;
    }
    QImage image(myRenderer.getBuffer(), width, height, QImage::Format_ARGB32);
    image.save(fileName.c_str(), "png");
  }
  else {
    QImage image(myRenderer.getBuffer(), width, height, QImage::Format_RGB888);
    image.save(fileName.c_str(), "png");
  }
}

void MainWindow::exportCurrentAsPNG() {
  ExportDialog dialog(this, false);
  dialog.exec();
  if(dialog.result()==QDialog::Rejected) return;

  statusBar->showMessage("Exporting current frame, please wait!");
  QColor c=dialog.getColor();
  SbVec2s size=glViewer->getSceneManager()->getViewportRegion().getWindowSize()*dialog.getScale();
  short width, height; size.getValue(width, height);
  SoOffscreenRenderer myRenderer(SbViewportRegion(width,height));
  exportAsPNG(myRenderer, dialog.getFileName().toStdString(), dialog.getTransparent(), c.red()/255.0, c.green()/255.0, c.blue()/255.0);
  statusBar->showMessage("Done", 2000);
}

void MainWindow::exportSequenceAsPNG() {
  ExportDialog dialog(this, true);
  dialog.exec();
  if(dialog.result()==QDialog::Rejected) return;
  double scale=dialog.getScale();
  bool transparent=dialog.getTransparent();
  QColor c=dialog.getColor();
  float red=c.red()/255.0, green=c.green()/255.0, blue=c.blue()/255.0;
  QString fileName=dialog.getFileName();
  if(!fileName.toUpper().endsWith(".PNG")) return;
  fileName=fileName.remove(fileName.length()-4,4);
  double speed=dialog.getSpeed();
  int startFrame=dialog.getStartFrame();
  int endFrame=dialog.getEndFrame();
  double fps=dialog.getFPS();

  if(speed/deltaTime/fps<1) {
    int ret=QMessageBox::warning(this, "Export PNG sequence",
      "Some video-frames would contain the same data,\n"
      "because the animation speed is to slow,\n"
      "and/or the framerate is to high\n"
      "and/or the time-intervall of the data files is to high.\n"
      "\n"
      "Continue anyway?", QMessageBox::Yes, QMessageBox::No);
    if(ret==QMessageBox::No) return;
  }
  int videoFrame=0;
  int lastVideoFrame=(int)(deltaTime*fps/speed*(endFrame-startFrame));
  SbVec2s size=glViewer->getSceneManager()->getViewportRegion().getWindowSize()*scale;
  short width, height; size.getValue(width, height);
  SoOffscreenRenderer myRenderer(SbViewportRegion(width,height));
int xx=0;
  for(int frame_=startFrame; frame_<=endFrame; frame_=(int)(speed/deltaTime/fps*++videoFrame+startFrame)) {
    statusBar->showMessage(QString("Exporting frame sequence, please wait! (%1\%)").arg(100.0*videoFrame/lastVideoFrame,0,'f',1));
    frame->setValue(frame_);
    exportAsPNG(myRenderer, QString("%1_%2.png").arg(fileName).arg(videoFrame, 6, 10, QChar('0')).toStdString(), transparent, red, green, blue);
  }
  statusBar->showMessage("Done", 2000);
}
