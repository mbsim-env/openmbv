#include "object.h"
#include <QtGui/QMenu>
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoDrawStyle.h>
#include <Inventor/actions/SoGetBoundingBoxAction.h>
#include "mainwindow.h"

using namespace std;

map<SoNode*,Object*> Object::objectMap;

Object::Object(TiXmlElement* element, H5::Group *h5Parent) : QTreeWidgetItem(), drawThisPath(true) {
  // h5 group
  if(element->Parent()->Type()==TiXmlNode::DOCUMENT)
    h5Group=h5Parent;
  else
    h5Group=new H5::Group(h5Parent->openGroup(element->Attribute("name")));

  // craete so basics (Separator)
  soSwitch=new SoSwitch;
  soSwitch->ref();
  soSwitch->whichChild.setValue(SO_SWITCH_ALL);
  soSep=new SoSeparator;
  soSwitch->addChild(soSep);

  // switch for bounding box
  soBBoxSwitch=new SoSwitch;
  MainWindow::sceneRootBBox->addChild(soBBoxSwitch);
  soBBoxSwitch->whichChild.setValue(SO_SWITCH_NONE);
  soBBoxSep=new SoSeparator;
  soBBoxSwitch->addChild(soBBoxSep);
  soBBoxTrans=new SoTranslation;
  soBBoxSep->addChild(soBBoxTrans);
  soBBox=new SoCube;
  soBBoxSep->addChild(soBBox);
  // register callback function on node change
  SoNodeSensor *sensor=new SoNodeSensor(nodeSensorCB, this);
  sensor->attach(soSep);

  // add to map for finding this object by the soSep SoNode
  objectMap.insert(pair<SoNode*, Object*>(soSep,this));

  setText(0, element->Attribute("name"));

  // GUI draw action
  draw=new QAction("Draw Object", 0);
  draw->setCheckable(true);
  draw->setChecked(true);
  connect(draw,SIGNAL(changed()),this,SLOT(drawSlot()));
  // GUI bbox action
  bbox=new QAction("Show Bounding Box", 0);
  bbox->setCheckable(true);
  connect(bbox,SIGNAL(changed()),this,SLOT(bboxSlot()));
}

// convenience: convert e.g. "[3;7;7.9]" to std::vector<double>(3,7,7.9)
vector<double> Object::toVector(string str) {
  for(int i=0; i<str.length(); i++)
    if(str[i]=='[' || str[i]==']' || str[i]==';') str[i]=' ';
  stringstream stream(str);
  double d;
  vector<double> ret;
  while(1) {
    stream>>d;
    if(stream.fail()) break;
    ret.push_back(d);
  }
  return ret;
}

// convenience: create frame so
SoSeparator* Object::soFrame(double size, double offset) {
  SoSeparator *sep=new SoSeparator;
  sep->ref();

  SoBaseColor *col;
  SoLineSet *line;

  // coordinates
  SoCoordinate3 *coord=new SoCoordinate3;
  sep->addChild(coord);
  coord->point.set1Value(0, -size/2+offset*size/2, 0, 0);
  coord->point.set1Value(1, +size/2+offset*size/2, 0, 0);
  coord->point.set1Value(2, 0, -size/2+offset*size/2, 0);
  coord->point.set1Value(3, 0, +size/2+offset*size/2, 0);
  coord->point.set1Value(4, 0, 0, -size/2+offset*size/2);
  coord->point.set1Value(5, 0, 0, +size/2+offset*size/2);

  // x-axis
  col=new SoBaseColor;
  col->rgb=SbColor(1, 0, 0);
  sep->addChild(col);
  line=new SoLineSet;
  line->startIndex.setValue(0);
  line->numVertices.setValue(2);
  sep->addChild(line);

  // y-axis
  col=new SoBaseColor;
  col->rgb=SbColor(0, 1, 0);
  sep->addChild(col);
  line=new SoLineSet;
  line->startIndex.setValue(2);
  line->numVertices.setValue(2);
  sep->addChild(line);

  // z-axis
  col=new SoBaseColor;
  col->rgb=SbColor(0, 0, 1);
  sep->addChild(col);
  line=new SoLineSet;
  line->startIndex.setValue(4);
  line->numVertices.setValue(2);
  sep->addChild(line);

  return sep;
}

QMenu* Object::createMenu() {
  QMenu *menu=new QMenu("Object Menu");
  QAction *dummy=new QAction("",menu);
  dummy->setEnabled(false);
  menu->addAction(dummy);
  menu->addSeparator()->setText("Properties form: Object");
  menu->addAction(draw);
  menu->addAction(bbox);
  return menu;
}

void Object::drawSlot() {
  if(draw->isChecked()) {
    soSwitch->whichChild.setValue(SO_SWITCH_ALL);
    setEnableRecursive(true);
  }
  else {
    soSwitch->whichChild.setValue(SO_SWITCH_NONE);
    setEnableRecursive(false);
  }
}

void Object::bboxSlot() {
  if(bbox->isChecked()) {
    soBBoxSwitch->whichChild.setValue(SO_SWITCH_ALL);
    soSep->touch(); // force an update
  }
  else
    soBBoxSwitch->whichChild.setValue(SO_SWITCH_NONE);
}

// set drawThisPath recursivly and colorisze the font
void Object::setEnableRecursive(bool enable) {
  if(enable && draw->isChecked() && (QTreeWidgetItem::parent()?((Object*)QTreeWidgetItem::parent())->drawThisPath:true)) {
    setForeground(0, QBrush(QColor(0,0,0))); // TODO color
    drawThisPath=true;
    for(int i=0; i<childCount(); i++)
      ((Object*)child(i))->setEnableRecursive(enable);
  }
  if(!enable) {
    setForeground(0, QBrush(QColor(128,128,128))); // TODO color
    drawThisPath=false;
    for(int i=0; i<childCount(); i++)
      ((Object*)child(i))->setEnableRecursive(enable);
  }
}

string Object::getPath() {
  if(QTreeWidgetItem::parent())
    return ((Object*)(QTreeWidgetItem::parent()))->getPath()+"/"+text(0).toStdString();
  else
    return text(0).toStdString();
}

QString Object::getInfo() {
  return QString("<b>Path:</b> %1<br/>").arg(getPath().c_str())+
         QString("<b>Class:</b> <img src=\"%1\" width=\"16\" height=\"16\"/> %2<br/>").arg(iconFile.c_str()).arg(metaObject()->className());
}

void Object::nodeSensorCB(void *data, SoSensor*) {
  Object *object=(Object*)data;
  if(object->bbox->isChecked()) {
    static SoGetBoundingBoxAction *bboxAction=new SoGetBoundingBoxAction(SbViewportRegion(0,0));
    bboxAction->apply(object->soSep);
    float x1,y1,z1,x2,y2,z2;
    bboxAction->getBoundingBox().getBounds(x1,y1,z1,x2,y2,z2);
    object->soBBox->width.setValue(x2-x1);
    object->soBBox->height.setValue(y2-y1);
    object->soBBox->depth.setValue(z2-z1);
    object->soBBoxTrans->translation.setValue((x1+x2)/2,(y1+y2)/2,(z1+z2)/2);
  }
}
