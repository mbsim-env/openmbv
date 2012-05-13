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
#include <QtGui/QGridLayout>
#include <Inventor/nodes/SoDrawStyle.h>
#include <Inventor/actions/SoGetBoundingBoxAction.h>
#include <Inventor/actions/SoSearchAction.h>
#include "mainwindow.h"
#include "compoundrigidbody.h"
#include "openmbvcppinterface/group.h"
#include "utils.h"

using namespace std;

set<Object*> Object::objects;

Object::Object(OpenMBV::Object* obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : QTreeWidgetItem(), drawThisPath(true), searchMatched(true) {
  object=obj;
  objects.insert(this);
  // parent item
  if(ind==-1 || ind>=parentItem->childCount())
    parentItem->addChild(this); // insert as last element
  else
    parentItem->insertChild(ind, this); // insert at position ind
  setFlags(flags() | Qt::ItemIsEditable);

  // enable or disable
  if((dynamic_cast<Object*>(parentItem)==0 && obj->getEnable()) || (obj->getEnable() && ((Object*)parentItem)->drawThisPath))
    drawThisPath=true;
  else
    drawThisPath=false;
  updateTextColor();
  
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
  CompoundRigidBody *crb=dynamic_cast<CompoundRigidBody*>(parentItem);
  if(!crb)
    nodeSensor->attach(soSep);
  else // for a Object in a CompoundRigidBody also the CompoundRigidBody must be honored on node changes
    nodeSensor->attach(crb->soSep);

  setText(0, obj->getName().c_str());

  clone=getClone();
  if(!clone)
    properties=new PropertyDialog(this);
  else {
    properties=clone->properties;
    properties->setParentObject(this);
  }

  //GUI editors
  QAction *deleteObject=new QAction(Utils::QIconCached(":/deleteobject.svg"), "Delete Object", this);
//MFMF multiedit  deleteObject->setObjectName("Group::deleteObject");
  connect(deleteObject,SIGNAL(activated()),this,SLOT(deleteObjectSlot()));
  properties->addContextAction(deleteObject);

  if(!clone) {
    BoolEditor *enableEditor=new BoolEditor(properties, Utils::QIconCached(":/drawobject.svg"), "Draw object");
    enableEditor->setOpenMBVParameter(object, &OpenMBV::Object::getEnable, &OpenMBV::Object::setEnable);
    properties->addPropertyAction(enableEditor->getAction()); // add this editor also to the context menu for convinience

    BoolEditor *boundingBoxEditor=new BoolEditor(properties, Utils::QIconCached(":/bbox.svg"), "Show bounding box");
    boundingBoxEditor->setOpenMBVParameter(object, &OpenMBV::Object::getBoundingBox, &OpenMBV::Object::setBoundingBox);
    properties->addPropertyAction(boundingBoxEditor->getAction()); // add this editor also to the context menu for convinience
  }
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
  if(!getClone())
    delete properties;
  objects.erase(this);
}

string Object::getPath() {
  return static_cast<Object*>(QTreeWidgetItem::parent())->getPath()+"/"+text(0).toStdString();
}

QString Object::getInfo() {
  return QString("<b>Path:</b> %1<br/>").arg(getPath().c_str())+
         QString("<b>Class:</b> <img src=\"%1\" width=\"16\" height=\"16\"/> %2").arg(iconFile.c_str()).arg(metaObject()->className());
}

void Object::nodeSensorCB(void *data, SoSensor*) {
  Object *object=(Object*)data;
  if(object->object->getBoundingBox()) {
    static SoGetBoundingBoxAction *bboxAction=new SoGetBoundingBoxAction(SbViewportRegion(0,0));
    bboxAction->apply(object->soSep);
    SoSearchAction sa;
    sa.setInterest(SoSearchAction::FIRST);
    sa.setNode(object->soSep);
    sa.apply(MainWindow::getInstance()->getSceneRoot());
    SoPath *p=sa.getPath();
    bboxAction->apply(p);
    float x1,y1,z1,x2,y2,z2;
    bboxAction->getBoundingBox().getBounds(x1,y1,z1,x2,y2,z2);
    object->soBBox->width.setValue(x2-x1);
    object->soBBox->height.setValue(y2-y1);
    object->soBBox->depth.setValue(z2-z1);
    object->soBBoxTrans->translation.setValue((x1+x2)/2,(y1+y2)/2,(z1+z2)/2);
  }
}

void Object::updateTextColor() {
  QPalette palette;
  if(drawThisPath) // active
    if(searchMatched)
      setForeground(0, palette.brush(QPalette::Active, QPalette::Text));
    else
      setForeground(0, QBrush(QColor(255,0,0)));
  else // inactive
    if(searchMatched)
      setForeground(0, palette.brush(QPalette::Disabled, QPalette::Text));
    else
      setForeground(0, QBrush(QColor(128,0,0)));
}

Object *Object::getClone() {
  set<Object*>::iterator it;
  for(it=objects.begin(); it!=objects.end(); it++)
    if(this!=(*it) && (*it)->object->getFullName()==object->getFullName())
      break;
  return it==objects.end()?NULL:*it;
}

void Object::deleteObjectSlot() {
  OpenMBV::Object *objPtr=object;
  // deleting an QTreeWidgetItem will remove the item from the tree (this is safe at any time)
  delete this; // from now no element should be accessed thats why we have saveed the obj member
  // if obj has a parent, remove obj from parent and delete obj
  objPtr->destroy(); // this does not use any member of Object, so we can call it after "detete this". We delete the OpenMBVCppInterface after the Object such that in the Object dtor the getPath is available
}
