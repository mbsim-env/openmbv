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
#include "compoundrigidbody.h"
#include "objectfactory.h"
#include "utils.h"
#include "QtGui/QLabel"
#include "QtGui/QPushButton"
#include "QtGui/QMessageBox"
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

CompoundRigidBody::CompoundRigidBody(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  crb=(OpenMBV::CompoundRigidBody*)obj;
  iconFile=":/compoundrigidbody.svg";
  setIcon(0, Utils::QIconCached(iconFile.c_str()));

  // expand or collapse
  setExpanded(crb->getExpand());

  // read XML
  vector<OpenMBV::RigidBody*> rb=crb->getRigidBodies();
  for(size_t i=0; i<rb.size(); i++)
    ObjectFactory(rb[i], this, soSep, -1);

  // GUI
  QAction *newObject=new QAction(Utils::QIconCached(":/newobject.svg"),"Create new RigidBody", this);
//MFMF multiedit  newObject->setObjectName("CompoundRigidBody::newRigidBody");
  connect(newObject,SIGNAL(activated()),this,SLOT(newRigidBodySlot()));
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
  static vector<Utils::FactoryElement> factory=boost::assign::list_of
    (Utils::FactoryElement(Utils::QIconCached(":/cube.svg"),          "Cube",          boost::factory<OpenMBV::Cube*>()))
    (Utils::FactoryElement(Utils::QIconCached(":/cuboid.svg"),        "Cuboid",        boost::factory<OpenMBV::Cuboid*>()))
    (Utils::FactoryElement(Utils::QIconCached(":/extrusion.svg"),     "Extrusion",     boost::factory<OpenMBV::Extrusion*>()))
    (Utils::FactoryElement(Utils::QIconCached(":/frame.svg"),         "Frame",         boost::factory<OpenMBV::Frame*>()))
    (Utils::FactoryElement(Utils::QIconCached(":/frustum.svg"),       "Frustum",       boost::factory<OpenMBV::Frustum*>()))
    (Utils::FactoryElement(Utils::QIconCached(":/invisiblebody.svg"), "Grid",          boost::factory<OpenMBV::Grid*>()))
    (Utils::FactoryElement(Utils::QIconCached(":/invisiblebody.svg"), "InvisibleBody", boost::factory<OpenMBV::InvisibleBody*>()))
    (Utils::FactoryElement(Utils::QIconCached(":/ivbody.svg"),        "IvBody",        boost::factory<OpenMBV::IvBody*>()))
    (Utils::FactoryElement(Utils::QIconCached(":/rotation.svg"),      "Rotation",      boost::factory<OpenMBV::Rotation*>()))
    (Utils::FactoryElement(Utils::QIconCached(":/sphere.svg"),        "Sphere",        boost::factory<OpenMBV::Sphere*>()))
  .to_container(factory);  

  vector<string> existingNames;
  for(unsigned int j=0; j<crb->getRigidBodies().size(); j++)
    existingNames.push_back(crb->getRigidBodies()[j]->getName());

  OpenMBV::Object *obj=Utils::createObjectEditor(factory, existingNames, "Create new RigidBody");
  if(obj==NULL) return;

  crb->addRigidBody(static_cast<OpenMBV::RigidBody*>(obj));
  ObjectFactory(obj, this, soSep, -1);
}
