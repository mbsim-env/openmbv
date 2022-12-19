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
#include "sphere.h"
#include <Inventor/nodes/SoSphere.h>
#include <Inventor/nodes/SoDrawStyle.h>
#include <vector>
#include "utils.h"
#include "openmbvcppinterface/sphere.h"
#include <QMenu>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

Sphere::Sphere(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  s=std::static_pointer_cast<OpenMBV::Sphere>(obj);
  iconFile="sphere.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // create so
  auto *sphere=new SoSphere;
  sphere->radius.setValue(s->getRadius());
  soSepRigidBody->addChild(sphere);
  // scale ref/localFrame
  refFrameScale->scaleFactor.setValue(2*s->getRadius()*s->getScaleFactor(),2*s->getRadius()*s->getScaleFactor(),2*s->getRadius()*s->getScaleFactor());
  localFrameScale->scaleFactor.setValue(2*s->getRadius()*s->getScaleFactor(),2*s->getRadius()*s->getScaleFactor(),2*s->getRadius()*s->getScaleFactor());

  // outline
  soSepRigidBody->addChild(soOutLineSwitch);
}

void Sphere::createProperties() {
  RigidBody::createProperties();

  if(!clone) {
    properties->updateHeader();
    // GUI editors
    auto *radiusEditor=new FloatEditor(properties, QIcon(), "Radius");
    radiusEditor->setRange(0, DBL_MAX);
    radiusEditor->setOpenMBVParameter(s, &OpenMBV::Sphere::getRadius, &OpenMBV::Sphere::setRadius);
  }
}

}
