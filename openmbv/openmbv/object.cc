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

namespace OpenMBVGUI {

// we use none signaling (quiet) NaN values for double in OpenMBV -> Throw compile error if these do not exist.
static_assert(numeric_limits<double>::has_quiet_NaN, "This platform does not support quiet NaN for double.");

set<Object*> Object::objects;

Object::Object(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : QTreeWidgetItem(), drawThisPath(true),
               properties(NULL), clone(NULL) {
  object=obj;
  objects.insert(this);
  // parent item
  if(ind==-1 || ind>=parentItem->childCount())
    parentItem->addChild(this); // insert as last element
  else
    parentItem->insertChild(ind, this); // insert at position ind
  setFlags(flags() | Qt::ItemIsEditable);

  // enable or disable
  QPalette palette;
  if((dynamic_cast<Object*>(parentItem)==0 && obj->getEnable()) || (obj->getEnable() && ((Object*)parentItem)->drawThisPath)) {
    drawThisPath=true;
    setData(0, Qt::UserRole, true); // UserRole is used by AbstractViewFilter for normal/grayed items
    setData(0, Qt::ForegroundRole, palette.brush(QPalette::Active, QPalette::Text));
  }
  else {
    drawThisPath=false;
    setData(0, Qt::UserRole, false); // UserRole is used by AbstractViewFilter for normal/grayed items
    setData(0, Qt::ForegroundRole, palette.brush(QPalette::Disabled, QPalette::Text));
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
  CompoundRigidBody *crb=dynamic_cast<CompoundRigidBody*>(parentItem);
  if(!crb)
    nodeSensor->attach(soSep);
  else // for a Object in a CompoundRigidBody also the CompoundRigidBody must be honored on node changes
    nodeSensor->attach(crb->soSep);

  // selected flag
  setSelected(object->getSelected());

  setText(0, obj->getName().c_str());

  clone=getClone();

  if(clone && clone->properties) {
    properties=clone->properties;
    properties->setParentObject(this);
    boundingBoxEditor=clone->boundingBoxEditor;
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

PropertyDialog *Object::getProperties() {
  if(!properties)
    createProperties();
  return properties;
}

void Object::createProperties() {
  properties=new PropertyDialog(this);

  //GUI editors
  QAction *deleteObject=new QAction(Utils::QIconCached("deleteobject.svg"), "Delete Object", properties);
  deleteObject->setObjectName("Group::deleteObject");
  connect(deleteObject,SIGNAL(triggered()),properties,SLOT(deleteObjectSlot()));
  properties->addContextAction(deleteObject);

  if(!clone) {
    BoolEditor *enableEditor=new BoolEditor(properties, Utils::QIconCached("drawobject.svg"), "Draw object", "Object::draw");
    enableEditor->setOpenMBVParameter(object, &OpenMBV::Object::getEnable, &OpenMBV::Object::setEnable);
    properties->addPropertyAction(enableEditor->getAction()); // add this editor also to the context menu for convinience

    boundingBoxEditor=new BoolEditor(properties, Utils::QIconCached("bbox.svg"), "Show bounding box", "Object::boundingBox", false);
    boundingBoxEditor->setOpenMBVParameter(object, &OpenMBV::Object::getBoundingBox, &OpenMBV::Object::setBoundingBox);
    properties->addPropertyAction(boundingBoxEditor->getAction()); // add this editor also to the context menu for convinience
    connect(boundingBoxEditor, SIGNAL(stateChanged(bool)), properties, SLOT(setBoundingBox(bool)));
  }
}

QString Object::getInfo() {
  return QString("<b>Path:</b> %1<br/>").arg(object->getFullName(true, true).c_str())+
         QString("<b>Class:</b> <img src=\"%1\" width=\"16\" height=\"16\"/> %2").
           arg((Utils::getIconPath()+"/"+getIconFile()).c_str()).
           arg(QString(metaObject()->className()).replace("OpenMBVGUI::", ""));  // remove the namespace
}

void Object::nodeSensorCB(void *data, SoSensor*) {
  Object *object=(Object*)data;
  if(object->object->getBoundingBox()) {
    SoSearchAction sa;
    sa.setInterest(SoSearchAction::FIRST);
    sa.setNode(object->soSep);
    sa.setSearchingAll(true);
    sa.apply(MainWindow::getInstance()->getSceneRoot());
    SoPath *p=sa.getPath();
    static SoGetBoundingBoxAction *bboxAction=new SoGetBoundingBoxAction(SbViewportRegion(0,0));
    bboxAction->apply(p);
    float x1,y1,z1,x2,y2,z2;
    bboxAction->getBoundingBox().getBounds(x1,y1,z1,x2,y2,z2);
    if(x1>x2 || y1>y2 || z1>z2)
      x1=x2=y1=y2=z1=z2=0;
    object->soBBox->width.setValue((x2-x1)*1.05);
    object->soBBox->height.setValue((y2-y1)*1.05);
    object->soBBox->depth.setValue((z2-z1)*1.05);
    object->soBBoxTrans->translation.setValue((x1+x2)/2,(y1+y2)/2,(z1+z2)/2);
  }
}

Object *Object::getClone() {
  set<Object*>::iterator it;
  for(it=objects.begin(); it!=objects.end(); it++)
    if(this!=(*it) && (*it)->object->getFullName(true)==object->getFullName(true))
      break;
  return it==objects.end()?NULL:*it;
}

void Object::deleteObjectSlot() {
  // deleting an QTreeWidgetItem will remove the item from the tree (this is safe at any time)
  delete this;
}

bool Object::getBoundingBox() {
  return object->getBoundingBox();
}

void Object::setBoundingBox(bool value) {
  soBBoxSwitch->whichChild.setValue(value?SO_SWITCH_ALL:SO_SWITCH_NONE);
  if(properties) {
    boundingBoxEditor->blockSignals(true);
    boundingBoxEditor->getAction()->setChecked(value);
    boundingBoxEditor->blockSignals(false);
  }
  else
    object->setBoundingBox(value);
  // update if true
  if(value)
    nodeSensorCB(this, NULL);
}

}
