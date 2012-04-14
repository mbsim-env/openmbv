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
#include "cuboid.h"
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoDrawStyle.h>
#include <vector>
#include "utils.h"
#include "openmbvcppinterface/cuboid.h"
#include <QMenu>
#include <cfloat>

using namespace std;

Cuboid::Cuboid(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  OpenMBV::Cuboid *c=(OpenMBV::Cuboid*)obj;
  iconFile=":/cuboid.svg";
  setIcon(0, Utils::QIconCached(iconFile.c_str()));

  // create so
  SoCube *cuboid=new SoCube;
  cuboid->width.setValue(c->getLength()[0]);
  cuboid->height.setValue(c->getLength()[1]);
  cuboid->depth.setValue(c->getLength()[2]);
  soSepRigidBody->addChild(cuboid);
  // scale ref/localFrame
  double size=min(c->getLength()[0],min(c->getLength()[1],c->getLength()[2]))*c->getScaleFactor();
  refFrameScale->scaleFactor.setValue(size,size,size);
  localFrameScale->scaleFactor.setValue(size,size,size);

  // outline
  soSepRigidBody->addChild(soOutLineSwitch);
  soOutLineSep->addChild(cuboid);

  // GUI
  lengthEditor=new Vec3fEditor(this, QIcon(), "Length (x, y, z)");
  lengthEditor->setRange(0, DBL_MAX);
  lengthEditor->setOpenMBVParameter(c, &OpenMBV::Cuboid::getLength, &OpenMBV::Cuboid::setLength);
}

QMenu* Cuboid::createMenu() {
  QMenu* menu=RigidBody::createMenu();
  menu->addSeparator()->setText("Properties from: Cuboid");
  menu->addAction(lengthEditor->getAction());
  return menu;
}
