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
#include "cube.h"
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoDrawStyle.h>
#include <vector>
#include "utils.h"
#include "openmbvcppinterface/cube.h"
#include <QMenu>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

Cube::Cube(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  OpenMBV::Cube *c=(OpenMBV::Cube*)obj;
  iconFile="cube.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // create so
  SoCube *cube=new SoCube;
  cube->width.setValue(c->getLength());
  cube->height.setValue(c->getLength());
  cube->depth.setValue(c->getLength());
  soSepRigidBody->addChild(cube);
  // scale ref/localFrame
  refFrameScale->scaleFactor.setValue(c->getLength()*c->getScaleFactor(),c->getLength()*c->getScaleFactor(),c->getLength()*c->getScaleFactor());
  localFrameScale->scaleFactor.setValue(c->getLength()*c->getScaleFactor(),c->getLength()*c->getScaleFactor(),c->getLength()*c->getScaleFactor());

  // outline
  soSepRigidBody->addChild(soOutLineSwitch);
  soOutLineSep->addChild(cube);

  // GUI editors
  if(!clone) {
    properties->updateHeader();
    FloatEditor *lengthEditor=new FloatEditor(properties, QIcon(), "Length");
    lengthEditor->setRange(0, DBL_MAX);
    lengthEditor->setOpenMBVParameter(c, &OpenMBV::Cube::getLength, &OpenMBV::Cube::setLength);
  }
}

}
