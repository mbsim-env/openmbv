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
#include "grid.h"
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoBaseColor.h>
#include "utils.h"
#include "openmbvcppinterface/grid.h"
#include <QMenu>
#include <cfloat>

namespace OpenMBVGUI {

Grid::Grid(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  g=std::static_pointer_cast<OpenMBV::Grid>(obj);
  iconFile="grid.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  auto *sep = new SoSeparator;

  auto *col=new SoBaseColor;
  col->rgb=SbColor(color, color, color);
  sep->addChild(col);

  double size=1.0;

  // coordinates
  auto * scale=new SoScale;
  sep->addChild(scale);
  scale->scaleFactor.setValue(size, size, size);
  auto *coord=new SoCoordinate3;
  sep->addChild(coord);
  int counter=0;
  for (unsigned int i=0; i<g->getYNumber(); i++) {
    coord->point.set1Value(counter++, -g->getXSize()/2., g->getYSize()/2.-double(i)*g->getYSize()/double(g->getYNumber()-1), 0);
    coord->point.set1Value(counter++, g->getXSize()/2., g->getYSize()/2.-double(i)*g->getYSize()/double(g->getYNumber()-1), 0);
  }
  for (unsigned int i=0; i<g->getXNumber(); i++) {
    coord->point.set1Value(counter++, -g->getXSize()/2.+double(i)*g->getXSize()/double(g->getXNumber()-1), g->getYSize()/2., 0);
    coord->point.set1Value(counter++, -g->getXSize()/2.+double(i)*g->getXSize()/double(g->getXNumber()-1), -g->getYSize()/2., 0);
  }
  
  for (int i=0; i<counter; i+=2) {
    auto * line=new SoLineSet;
    line->startIndex.setValue(i);
    line->numVertices.setValue(2);
    sep->addChild(line);
  }

  // create so
  soSepRigidBody->addChild(sep);
}

void Grid::createProperties() {
  RigidBody::createProperties();

  // GUI editors
  if(!clone) {
    properties->updateHeader();
    auto *xSizeEditor=new FloatEditor(properties, QIcon(), "Grid x-size");
    xSizeEditor->setRange(0, DBL_MAX);
    xSizeEditor->setOpenMBVParameter(g, &OpenMBV::Grid::getXSize, &OpenMBV::Grid::setXSize);

    auto *ySizeEditor=new FloatEditor(properties, QIcon(), "Grid y-size");
    ySizeEditor->setRange(0, DBL_MAX);
    ySizeEditor->setOpenMBVParameter(g, &OpenMBV::Grid::getYSize, &OpenMBV::Grid::setYSize);

    auto *nxEditor=new IntEditor(properties, QIcon(), "Number of x-grids");
    nxEditor->setRange(0, INT_MAX);
    nxEditor->setOpenMBVParameter(g, &OpenMBV::Grid::getXNumber, &OpenMBV::Grid::setXNumber);

    auto *nyEditor=new IntEditor(properties, QIcon(), "Number of y-grids");
    nyEditor->setRange(0, INT_MAX);
    nyEditor->setOpenMBVParameter(g, &OpenMBV::Grid::getYNumber, &OpenMBV::Grid::setYNumber);
  }
}

}
