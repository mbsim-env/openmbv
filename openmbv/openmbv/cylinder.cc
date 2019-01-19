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
#include "cylinder.h"
#include <Inventor/nodes/SoCylinder.h>
#include <Inventor/nodes/SoDrawStyle.h>
#include <vector>
#include "utils.h"
#include "openmbvcppinterface/cylinder.h"
#include <QMenu>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

Cylinder::Cylinder(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  c=std::static_pointer_cast<OpenMBV::Cylinder>(obj);
  iconFile="cylinder.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // create so
  auto *cylinder=new SoCylinder;
  cylinder->radius.setValue(c->getRadius());
  cylinder->height.setValue(c->getHeight());
  soSepRigidBody->addChild(cylinder);
  // scale ref/localFrame
  double size=min(c->getRadius(),c->getHeight())*c->getScaleFactor();
  refFrameScale->scaleFactor.setValue(size,size,size);
  localFrameScale->scaleFactor.setValue(size,size,size);

  // outline
  soSepRigidBody->addChild(soOutLineSwitch);
}

void Cylinder::createProperties() {
  RigidBody::createProperties();

  if(!clone) {
    properties->updateHeader();
    // GUI editors
    FloatEditor *radiusEditor=new FloatEditor(properties, QIcon(), "Radius");
    radiusEditor->setRange(0, DBL_MAX);
    radiusEditor->setOpenMBVParameter(c, &OpenMBV::Cylinder::getRadius, &OpenMBV::Cylinder::setRadius);
    FloatEditor *heightEditor=new FloatEditor(properties, QIcon(), "Height");
    heightEditor->setRange(0, DBL_MAX);
    heightEditor->setOpenMBVParameter(c, &OpenMBV::Cylinder::getHeight, &OpenMBV::Cylinder::setHeight);
  }
}

}
