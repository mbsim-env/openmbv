/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

  This library is free software; you can redistribute it and/or 
  modify it under the terms of the GNU Lesser General Public 
  License as published by the Free Software Foundation; either 
  version 2.1 of the License, or (at your option) any later version. 
   
  This library is distributed in the hope that it will be useful, 
  but WITHOUT ANY WARRANTY; without even the implied warranty of 
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
  Lesser General Public License for more details. 
   
  You should have received a copy of the GNU Lesser General Public 
  License along with this library; if not, write to the Free Software 
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
*/

#include "config.h"
#include "object.h"
#include <QMenu>
#include <QApplication>
#include <QGridLayout>
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

map<SoNode*, Object*> Object::objectMap;

Object::Object(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) :  drawThisPath(true),
               properties(nullptr), clone(nullptr) {
  bool isClone=false;
  if(ind<=-2) { // a ind <=-2 means that this is a cloned object
    isClone=true; // marke this object as a clone
    ind=-ind-3; // fix the index
  }
  object=obj;
  // parent item
  if(ind==-1 || ind>=parentItem->childCount())
    parentItem->addChild(this); // insert as last element
  else
    parentItem->insertChild(ind, this); // insert at position ind
  setFlags(flags() | Qt::ItemIsEditable);

  // enable or disable
  QPalette palette;
  if((dynamic_cast<Object*>(parentItem)==nullptr && obj->getEnable()) || (obj->getEnable() && ((Object*)parentItem)->drawThisPath)) {
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
  objectMap.emplace(soSep, this);
  soSep->renderCaching.setValue(SoSeparator::OFF); // a object at least moves (so disable caching)
  soSwitch->addChild(soSep);

  // switch for bounding box
  soBBoxSwitch=new SoSwitch;
  MainWindow::getInstance()->getSceneRootBBox()->addChild(soBBoxSwitch);
  soBBoxSwitch->whichChild.setValue(obj->getBoundingBox()?SO_SWITCH_ALL:SO_SWITCH_NONE);
  soBBoxSep=new SoSeparator;
  soBBoxSwitch->addChild(soBBoxSep);
  soBBoxTrans=new SoMatrixTransform;
  soBBoxSep->addChild(soBBoxTrans);
  soBBoxSep->addChild(MainWindow::getInstance()->bboxColor);
  soBBoxSep->addChild(MainWindow::getInstance()->bboxDrawStyle);
  soBBox=new SoCube;
  soBBoxSep->addChild(soBBox);
  // register callback function on node change
  nodeSensor=new SoNodeSensor(nodeSensorCB, this);
  auto *crb=dynamic_cast<CompoundRigidBody*>(parentItem);
  if(!crb)
    nodeSensor->attach(soSep);
  else // for a Object in a CompoundRigidBody also the CompoundRigidBody must be honored on node changes
    nodeSensor->attach(crb->soSep);

  setText(0, obj->getName().c_str());

  clone=nullptr;
  if(isClone) { // the below code is quite expensive, that's why we do it only if we know that this object is a clone
    map<SoNode*, Object*>::iterator it;
    for(it=objectMap.begin(); it!=objectMap.end(); it++)
      if(this!=it->second && it->second->object->getFullName()==object->getFullName())
        break;
    clone = it==objectMap.end()?NULL:it->second;
  }

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
  if(!isCloneToBeDeleted)
    delete properties;
  for(auto [node,obj] : objectMap)
    if(obj==this) {
      objectMap.erase(node);
      break;
    }
}

PropertyDialog *Object::getProperties() {
  if(!properties)
    createProperties();
  return properties;
}

void Object::createProperties() {
  properties=new PropertyDialog(this);

  //GUI editors
  auto *deleteObject=new QAction(Utils::QIconCached("deleteobject.svg"), "Delete Object", properties);
  deleteObject->setObjectName("Group::deleteObject");
  connect(deleteObject,&QAction::triggered,this,[this](){
    static_cast<Object*>(properties->getParentObject())->deleteObjectSlot();
  });
  properties->addContextAction(deleteObject);

  if(!clone) {
    auto *enableEditor=new BoolEditor(properties, Utils::QIconCached("drawobject.svg"), "Draw object", "Object::draw");
    enableEditor->setOpenMBVParameter(object, &OpenMBV::Object::getEnable, &OpenMBV::Object::setEnable);
    properties->addPropertyAction(enableEditor->getAction()); // add this editor also to the context menu for convinience

    boundingBoxEditor=new BoolEditor(properties, Utils::QIconCached("bbox.svg"), "Show bounding box", "Object::boundingBox", false);
    boundingBoxEditor->setOpenMBVParameter(object, &OpenMBV::Object::getBoundingBox, &OpenMBV::Object::setBoundingBox);
    properties->addPropertyAction(boundingBoxEditor->getAction()); // add this editor also to the context menu for convinience
    connect(boundingBoxEditor, &BoolEditor::stateChanged, this, [this](bool b){
      static_cast<Object*>(properties->getParentObject())->setBoundingBox(b);
    });
  }
}

QString Object::getInfo() {
  return QString("<b>Path:</b> %1<br/>").arg(object->getFullName().c_str())+
         QString(R"(<b>Class:</b> <img src="%1" width="16" height="16"/> %2)").
           arg((Utils::getIconPath()+"/"+getIconFile()).c_str()).
           arg(QString(metaObject()->className()).replace("OpenMBVGUI::", ""));  // remove the namespace
}

void Object::nodeSensorCB(void *data, SoSensor*) {
  auto *object=(Object*)data;
  if(object->drawBoundingBox()) {
    SoSearchAction sa;
    sa.setInterest(SoSearchAction::FIRST);
    sa.setNode(object->soSep);
    sa.setSearchingAll(true);
    sa.apply(MainWindow::getInstance()->getSceneRoot());
    SoPath *p=sa.getPath();
    static auto *bboxAction=new SoGetBoundingBoxAction(SbViewportRegion(0,0));
    bboxAction->apply(p);
    auto bbox=bboxAction->getXfBoundingBox();
    float x,y,z;
    bbox.getSize(x,y,z);
    object->soBBox->width.setValue(x*1.05);
    object->soBBox->height.setValue(y*1.05);
    object->soBBox->depth.setValue(z*1.05);
    const SbMatrix &m=bbox.getTransform();
    SbVec3f t, s;
    SbRotation r, so;
    m.getTransform(t,r,s,so);
    SbMatrix m2;
    m2.setTransform(bbox.getCenter(),r,s,so);
    object->soBBoxTrans->matrix.setValue(m2);
  }
}

void Object::deleteObjectSlot() {
  // deleting an QTreeWidgetItem will remove the item from the tree (this is safe at any time)
  delete this;
  
  MainWindow::getInstance()->updateBackgroundNeeded();
}

void Object::replaceBBoxHighlight() {
  soBBoxSwitch->whichChild.setValue(drawBoundingBox()?SO_SWITCH_ALL:SO_SWITCH_NONE);
  if(highlight) {
    int idx = soBBoxSep->findChild(MainWindow::getInstance()->bboxColor);
    if(idx>=0) soBBoxSep->replaceChild(idx, MainWindow::getInstance()->highlightColor);
    idx = soBBoxSep->findChild(MainWindow::getInstance()->bboxDrawStyle);
    if(idx>=0) soBBoxSep->replaceChild(idx, MainWindow::getInstance()->highlightDrawStyle);
  }
  else {
    int idx = soBBoxSep->findChild(MainWindow::getInstance()->highlightColor);
    if(idx>=0) soBBoxSep->replaceChild(idx, MainWindow::getInstance()->bboxColor);
    idx = soBBoxSep->findChild(MainWindow::getInstance()->highlightDrawStyle);
    if(idx>=0) soBBoxSep->replaceChild(idx, MainWindow::getInstance()->bboxDrawStyle);
  }
}

void Object::setBoundingBox(bool value) {
  if(properties) {
    boundingBoxEditor->blockSignals(true);
    boundingBoxEditor->getAction()->setChecked(value);
    boundingBoxEditor->blockSignals(false);
  }
  else
    object->setBoundingBox(value);
  replaceBBoxHighlight();
  // update boundingbox action if true
  if(value)
    nodeSensorCB(this, nullptr);
}

void Object::setHighlight(bool value) {
  highlight=value;
  replaceBBoxHighlight();
  // update boundingbox action if true
  if(value)
    nodeSensorCB(this, nullptr);
}

}
