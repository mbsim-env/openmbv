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
#include "object.h"
#include <QtGui/QMenu>
#include <QtGui/QApplication>
#include <Inventor/nodes/SoDrawStyle.h>
#include <Inventor/actions/SoGetBoundingBoxAction.h>
#include "mainwindow.h"
#include "openmbvcppinterface/group.h"
#include "compoundrigidbody.h"
#include "utils.h"

using namespace std;

map<SoNode*,Object*> Object::objectMap;

Object::Object(OpenMBV::Object* obj, QTreeWidgetItem *parentItem, SoGroup *soParent) : QTreeWidgetItem(), drawThisPath(true), searchMatched(true) {
  object=obj;
  if(dynamic_cast<CompoundRigidBody*>(parentItem)==0) {
    // parent item
    parentItem->addChild(this);

    // enable or disable
    if((dynamic_cast<Object*>(parentItem)==0 && obj->getEnable()) || (obj->getEnable() && ((Object*)parentItem)->drawThisPath))
      drawThisPath=true;
    else
      drawThisPath=false;
    updateTextColor();
  }
  
  // craete so basics (Separator)
  soSwitch=new SoSwitch;
  soParent->addChild(soSwitch); // parent so
  soSwitch->ref();
  soSwitch->whichChild.setValue(obj->getEnable()?SO_SWITCH_ALL:SO_SWITCH_NONE);
  soSep=new SoSeparator;
  soSep->renderCaching.setValue(SoSeparator::OFF); // a object at least moves (so disable caching)
  soSwitch->addChild(soSep);

  // switch for bounding box
  soBBoxSwitch=new SoSwitch;
  MainWindow::getInstance()->getSceneRootBBox()->addChild(soBBoxSwitch);
  soBBoxSwitch->whichChild.setValue(obj->getBoundingBox()?SO_SWITCH_ALL:SO_SWITCH_NONE);
  soBBoxSep=new SoSeparator;
  soBBoxSwitch->addChild(soBBoxSep);
  soBBoxTrans=new SoTranslation;
  soBBoxSep->addChild(soBBoxTrans);
  soBBox=new SoCube;
  soBBoxSep->addChild(soBBox);
  // register callback function on node change
  nodeSensor=new SoNodeSensor(nodeSensorCB, this);
  nodeSensor->attach(soSep);

  // add to map for finding this object by the soSep SoNode
  if(dynamic_cast<CompoundRigidBody*>(parentItem)==0)
    objectMap.insert(pair<SoNode*, Object*>(soSep,this));

  setText(0, obj->getName().c_str());

  // GUI draw action
  draw=new QAction(Utils::QIconCached(":/drawobject.svg"),"Draw Object", this);
  draw->setCheckable(true);
  draw->setChecked(obj->getEnable());
  draw->setObjectName("Object::draw");
  connect(draw,SIGNAL(changed()),this,SLOT(drawSlot()));
  // GUI bbox action
  bbox=new QAction(Utils::QIconCached(":/bbox.svg"),"Show Bounding Box", this);
  bbox->setCheckable(true);
  bbox->setChecked(obj->getBoundingBox());
  bbox->setObjectName("Object::bbox");
  connect(bbox,SIGNAL(changed()),this,SLOT(bboxSlot()));
}

Object::~Object() {
  delete nodeSensor;
}

QMenu* Object::createMenu() {
  QMenu *menu=new QMenu("Object Menu");
  QAction *dummy=new QAction("",menu);
  dummy->setEnabled(false);
  menu->addAction(dummy);
  menu->addSeparator()->setText("Properties from: Object");
  menu->addAction(draw);
  menu->addAction(bbox);
  return menu;
}

void Object::drawSlot() {
  if(draw->isChecked()) {
    soSwitch->whichChild.setValue(SO_SWITCH_ALL);
    object->setEnable(true);
    setEnableRecursive(true);
  }
  else {
    soSwitch->whichChild.setValue(SO_SWITCH_NONE);
    object->setEnable(false);
    setEnableRecursive(false);
  }
}

void Object::bboxSlot() {
  if(bbox->isChecked()) {
    soBBoxSwitch->whichChild.setValue(SO_SWITCH_ALL);
    object->setBoundingBox(true);
    soSep->touch(); // force an update
  }
  else {
    soBBoxSwitch->whichChild.setValue(SO_SWITCH_NONE);
    object->setBoundingBox(false);
  }
}

// set drawThisPath recursivly and colorisze the font
void Object::setEnableRecursive(bool enable) {
  if(enable && draw->isChecked() && (QTreeWidgetItem::parent()?((Object*)QTreeWidgetItem::parent())->drawThisPath:true)) {
    drawThisPath=true;
    for(int i=0; i<childCount(); i++)
      ((Object*)child(i))->setEnableRecursive(enable);
  }
  if(!enable) {
    drawThisPath=false;
    for(int i=0; i<childCount(); i++)
      ((Object*)child(i))->setEnableRecursive(enable);
  }
  updateTextColor();
}

string Object::getPath() {
  if(QTreeWidgetItem::parent())
    return ((Object*)(QTreeWidgetItem::parent()))->getPath()+"/"+text(0).toStdString();
  else
    return text(0).toStdString();
}

QString Object::getInfo() {
  return QString("<b>Path:</b> %1<br/>").arg(getPath().c_str())+
         QString("<b>Class:</b> <img src=\"%1\" width=\"16\" height=\"16\"/> %2").arg(iconFile.c_str()).arg(metaObject()->className());
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

void Object::updateTextColor() {
  if(drawThisPath)
    if(searchMatched)
      setForeground(0, QBrush(QApplication::style()->standardPalette().color(QPalette::Active, QPalette::Text)));
    else
      setForeground(0, QBrush(QColor(255,0,0)));
  else
    if(searchMatched)
      setForeground(0, QBrush(QApplication::style()->standardPalette().color(QPalette::Disabled, QPalette::Text)));
    else
      setForeground(0, QBrush(QColor(128,0,0)));
}
