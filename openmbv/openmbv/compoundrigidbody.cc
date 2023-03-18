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
#include "compoundrigidbody.h"
#include "objectfactory.h"
#include "mainwindow.h"
#include "utils.h"
#include "QLabel"
#include "QPushButton"
#include "QMessageBox"
#include "openmbvcppinterface/arrow.h"
#include "openmbvcppinterface/coilspring.h"
#include "openmbvcppinterface/nurbsdisk.h"
#include "openmbvcppinterface/cube.h"
#include "openmbvcppinterface/cuboid.h"
#include "openmbvcppinterface/extrusion.h"
#include "openmbvcppinterface/frame.h"
#include "openmbvcppinterface/frustum.h"
#include "openmbvcppinterface/grid.h"
#include "openmbvcppinterface/invisiblebody.h"
#include "openmbvcppinterface/ivbody.h"
#include "openmbvcppinterface/rotation.h"
#include "openmbvcppinterface/sphere.h"

using namespace std;

namespace OpenMBVGUI {

CompoundRigidBody::CompoundRigidBody(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  crb=std::static_pointer_cast<OpenMBV::CompoundRigidBody>(obj);
  iconFile="compoundrigidbody.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // expand or collapse
  setExpanded(crb->getExpand());

  // read XML
  vector<std::shared_ptr<OpenMBV::RigidBody> > rb=crb->getRigidBodies();
  for(const auto & i : rb)
    ObjectFactory::create(i, this, soSep, -1);
}

void CompoundRigidBody::createProperties() {
  RigidBody::createProperties();

  // GUI
  auto *newObject=new QAction(Utils::QIconCached("newobject.svg"),"Create new RigidBody", properties);
  connect(newObject,&QAction::triggered,this,[this](){
    static_cast<CompoundRigidBody*>(properties->getParentObject())->newRigidBodySlot();
  });
  properties->addContextAction(newObject);

  if(!clone) {
    properties->updateHeader();
  }
}

QString CompoundRigidBody::getInfo() {
  return RigidBody::getInfo()+
         QString("<hr width=\"10000\"/>")+
         QString("<b>Number of children:</b> %1").arg(childCount());
}

void CompoundRigidBody::newRigidBodySlot() {
  static vector<Utils::FactoryElement> factory={
    {Utils::FactoryElement(Utils::QIconCached("cube.svg"),          "Cube",          Utils::factory<OpenMBV::Cube>())},
    {Utils::FactoryElement(Utils::QIconCached("cuboid.svg"),        "Cuboid",        Utils::factory<OpenMBV::Cuboid>())},
    {Utils::FactoryElement(Utils::QIconCached("extrusion.svg"),     "Extrusion",     Utils::factory<OpenMBV::Extrusion>())},
    {Utils::FactoryElement(Utils::QIconCached("frame.svg"),         "Frame",         Utils::factory<OpenMBV::Frame>())},
    {Utils::FactoryElement(Utils::QIconCached("frustum.svg"),       "Frustum",       Utils::factory<OpenMBV::Frustum>())},
    {Utils::FactoryElement(Utils::QIconCached("invisiblebody.svg"), "Grid",          Utils::factory<OpenMBV::Grid>())},
    {Utils::FactoryElement(Utils::QIconCached("invisiblebody.svg"), "InvisibleBody", Utils::factory<OpenMBV::InvisibleBody>())},
    {Utils::FactoryElement(Utils::QIconCached("ivbody.svg"),        "IvBody",        Utils::factory<OpenMBV::IvBody>())},
    {Utils::FactoryElement(Utils::QIconCached("rotation.svg"),      "Rotation",      Utils::factory<OpenMBV::Rotation>())},
    {Utils::FactoryElement(Utils::QIconCached("sphere.svg"),        "Sphere",        Utils::factory<OpenMBV::Sphere>())}
  };

  vector<string> existingNames;
  for(auto & j : crb->getRigidBodies())
    existingNames.push_back(j->getName());

  std::shared_ptr<OpenMBV::Object> obj=Utils::createObjectEditor(factory, existingNames, "Create new RigidBody");
  if(!obj) return;

  crb->addRigidBody(std::static_pointer_cast<OpenMBV::RigidBody>(obj));
  ObjectFactory::create(obj, this, soSep, -1);

  // apply object filter
  MainWindow::getInstance()->objectListFilter->applyFilter();
}

double CompoundRigidBody::update() {
  if(rigidBody->getRows()==0) return 0; // do nothing for environement objects

  // call the normal update for a RigidBody
  double t=RigidBody::update();

  // update the color for children
  int frame=MainWindow::getInstance()->getFrame()->getValue();
  vector<double> data=rigidBody->getRow(frame);
  for(int i=0; i<childCount(); i++) {
    auto *childRB=static_cast<RigidBody*>(child(i));
    if(childRB->diffuseColor[0]<0)
      childRB->setColor(data[7]);
  }

  return t;
}

}
