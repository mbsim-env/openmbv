#include "object.h"
#include <QtGui/QMenu>
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoBaseColor.h>

using namespace std;

SoSFUInt32 *Object::frame;

Object::Object(TiXmlElement* element) : QTreeWidgetItem(), drawThisPath(true) {
  // initialize global frame field
  frame=(SoSFUInt32*)SoDB::createGlobalField("frame",SoSFUInt32::getClassTypeId());

  // craete so basics (Separator)
  soSwitch=new SoSwitch;
  soSwitch->ref();
  soSwitch->whichChild.setValue(SO_SWITCH_ALL);
  soSep=new SoSeparator;
  soSwitch->addChild(soSep);

  setText(0, element->Attribute("name"));

  // GUI
  draw=new QAction("Draw Object", 0);
  draw->setCheckable(true);
  draw->setChecked(true);
  connect(draw,SIGNAL(changed()),this,SLOT(drawSlot()));
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
  QAction *type=new QAction(QString("Class: ")+metaObject()->className(), menu);
  type->setEnabled(false);
  menu->addAction(type);
  menu->addSeparator();
  menu->addAction(draw);
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
