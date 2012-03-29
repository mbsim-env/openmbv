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
#include <Inventor/actions/SoSearchAction.h>
#include "mainwindow.h"
#include "openmbvcppinterface/group.h"
#include "compoundrigidbody.h"
#include "utils.h"

using namespace std;

Object::Object(OpenMBV::Object* obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : QTreeWidgetItem(), drawThisPath(true), searchMatched(true) {
  object=obj;
  if(dynamic_cast<CompoundRigidBody*>(parentItem)==0) {
    // parent item
    if(ind==-1 || ind>=parentItem->childCount())
      parentItem->addChild(this); // insert as last element
    else
      parentItem->insertChild(ind, this); // insert at position ind

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

  setText(0, obj->getName().c_str());

  // GUI draw/bbox editor
  draw=new BoolEditor(this, Utils::QIconCached(":/drawobject.svg"), "Draw Object");
  draw->setOpenMBVParameter(obj, &OpenMBV::Object::getEnable, &OpenMBV::Object::setEnable);
  connect(draw->getAction(),SIGNAL(toggled(bool)),this,SLOT(setEnableRecursive(bool))); // a special action is required for enable/disable
  bbox=new BoolEditor(this, Utils::QIconCached(":/bbox.svg"), "Show Bounding Box");
  bbox->setOpenMBVParameter(obj, &OpenMBV::Object::getBoundingBox, &OpenMBV::Object::setBoundingBox);
  connect(bbox->getAction(),SIGNAL(changed()),this,SLOT(bboxSlot()));
}

Object::~Object() {
  // delete scene graph
  SoSearchAction sa;
  sa.setInterest(SoSearchAction::FIRST);
  sa.setNode(soSwitch);
  sa.apply(MainWindow::getInstance()->getSceneRoot());
  SoPath *p=sa.getPath();
  if(p) ((SoSwitch*)p->getNodeFromTail(1))->removeChild(soSwitch);
  // delete bbox scene graph
  sa.setNode(soBBoxSwitch);
  sa.apply(MainWindow::getInstance()->getSceneRootBBox());
  p=sa.getPath();
  if(p) ((SoSwitch*)p->getNodeFromTail(1))->removeChild(soBBoxSwitch);
  // delete the rest
  delete nodeSensor;
  soSwitch->unref();
}

QMenu* Object::createMenu() {
  QMenu *menu=new QMenu("Object Menu");
  QAction *dummy=new QAction("",menu);
  dummy->setEnabled(false);
  menu->addAction(dummy);
  menu->addSeparator()->setText("Properties from: Object");
  menu->addAction(draw->getAction());
  menu->addAction(bbox->getAction());
  return menu;
}

void Object::bboxSlot() {
  soSep->touch(); // force an update
}

// set drawThisPath recursivly and colorisze the font
void Object::setEnableRecursive(bool enable) {
  if(enable && draw->getAction()->isChecked() && (QTreeWidgetItem::parent()?((Object*)QTreeWidgetItem::parent())->drawThisPath:true)) {
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
  if(object->bbox->getAction()->isChecked()) {
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
