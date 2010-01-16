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

Grid::Grid(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : RigidBody(element, h5Parent, parentItem, soParent) {
  iconFile=":/grid.svg";
  setIcon(0, QIconCached(iconFile.c_str()));

//  // read XML
  TiXmlElement *e=element->FirstChildElement(OPENMBVNS"xSize");
  double dx=toVector(e->GetText())[0];
  e=element->FirstChildElement(OPENMBVNS"ySize");
  double dy=toVector(e->GetText())[0];
  e=element->FirstChildElement(OPENMBVNS"nx");
  int nx=int(toVector(e->GetText())[0]+.5);
  e=element->FirstChildElement(OPENMBVNS"ny");
  int ny=int(toVector(e->GetText())[0]+.5);

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
  for (int i=0; i<ny; i++) {
    coord->point.set1Value(counter++, -dx/2., dy/2.-double(i)*dy/double(ny-1), 0);
    coord->point.set1Value(counter++, dx/2., dy/2.-double(i)*dy/double(ny-1), 0);
  }
  for (int i=0; i<nx; i++) {
    coord->point.set1Value(counter++, -dx/2.+double(i)*dx/double(nx-1), dy/2., 0);
    coord->point.set1Value(counter++, -dx/2.+double(i)*dx/double(nx-1), -dy/2., 0);
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
