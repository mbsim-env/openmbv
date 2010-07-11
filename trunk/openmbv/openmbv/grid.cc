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
#include "grid.h"
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoBaseColor.h>
#include "utils.h"
#include "openmbvcppinterface/grid.h"

Grid::Grid(OpenMBV::Object *obj, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : RigidBody(obj, h5Parent, parentItem, soParent) {
  OpenMBV::Grid *g=(OpenMBV::Grid*)obj;
  iconFile=":/grid.svg";
  setIcon(0, Utils::QIconCached(iconFile.c_str()));

  SoSeparator *sep = new SoSeparator;

  SoBaseColor *col=new SoBaseColor;
  col->rgb=SbColor(color, color, color);
  sep->addChild(col);

  double size=1.0;

  // coordinates
  SoScale * scale=new SoScale;
  sep->addChild(scale);
  scale->scaleFactor.setValue(size, size, size);
  SoCoordinate3 *coord=new SoCoordinate3;
  sep->addChild(coord);
  int counter=0;
  for (int i=0; i<g->getYNumber(); i++) {
    coord->point.set1Value(counter++, -g->getXSize()/2., g->getYSize()/2.-double(i)*g->getYSize()/double(g->getYNumber()-1), 0);
    coord->point.set1Value(counter++, g->getXSize()/2., g->getYSize()/2.-double(i)*g->getYSize()/double(g->getYNumber()-1), 0);
  }
  for (int i=0; i<g->getXNumber(); i++) {
    coord->point.set1Value(counter++, -g->getXSize()/2.+double(i)*g->getXSize()/double(g->getXNumber()-1), g->getYSize()/2., 0);
    coord->point.set1Value(counter++, -g->getXSize()/2.+double(i)*g->getXSize()/double(g->getXNumber()-1), -g->getYSize()/2., 0);
  }
  
  for (int i=0; i<counter; i+=2) {
    SoLineSet * line=new SoLineSet;
    line->startIndex.setValue(i);
    line->numVertices.setValue(2);
    sep->addChild(line);
  }

  // create so
  soSepRigidBody->addChild(sep);
}
