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

using namespace std;

Cube::Cube(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : RigidBody(element, h5Parent, parentItem, soParent) {
  iconFile=":/cube.svg";
  setIcon(0, QIcon(iconFile.c_str()));

  // read XML
  TiXmlElement *e=element->FirstChildElement(OPENMBVNS"length");
  double length=toVector(e->GetText())[0];

  // create so
  SoCube *cube=new SoCube;
  cube->width.setValue(length);
  cube->height.setValue(length);
  cube->depth.setValue(length);
  soSep->addChild(cube);
  // scale ref/localFrame
  refFrameScale->scaleFactor.setValue(length,length,length);
  localFrameScale->scaleFactor.setValue(length,length,length);

  // outline
  soSep->addChild(soOutLineSwitch);
  soOutLineSep->addChild(cube);
}
