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

  double size=min(c->getRadius(),c->getHeight())*c->getScaleFactor();
  refFrameScale->scaleFactor.setValue(size,size,size);
  localFrameScale->scaleFactor.setValue(size,size,size);

  // outline
  const int num=128;
  float radius = c->getRadius();
  float y = c->getHeight()*0.5f;
  float delta = (2.0*M_PI)/(double)num;
  float angle = 0.0f;
  float pts[2*num][3];
  for (unsigned int i=0; i<num; i++) {
    pts[i][0] = -sin(angle)*radius;
    pts[i][1] = y;
    pts[i][2] = -cos(angle)*radius;
    pts[num+i][0] = pts[i][0];
    pts[num+i][1] = -y;
    pts[num+i][2] = pts[i][2];
    angle += delta;
  }
  int index[2*num+4];
  int k=0;
  for(unsigned int i=0; i<num; i++)
    index[k++] = i;
  index[k++] = 0;
  index[k++] = -1;
  for(unsigned int i=num; i<2*num; i++)
    index[k++] = i;
  index[k++] = num;
  index[k++] = -1;

  int indexf[13*(num-1)+5];
  k=0;
  for(unsigned int i=0; i<num-1; i++) {
    indexf[k++] = 0;
    indexf[k++] = i;
    indexf[k++] = i+1;
    indexf[k++] = -1;
  }
  for(unsigned int i=num; i<2*num-1; i++) {
    indexf[k++] = num;
    indexf[k++] = i+1;
    indexf[k++] = i;
    indexf[k++] = -1;
  }
  for(unsigned int i=0; i<num-1; i++) {
    indexf[k++] = num+i;
    indexf[k++] = num+i+1;
    indexf[k++] = i+1;
    indexf[k++] = i;
    indexf[k++] = -1;
  }
  indexf[k++] = 2*num-1;
  indexf[k++] = num;
  indexf[k++] = 0;
  indexf[k++] = num-1;
  indexf[k++] = -1;

  auto *points = new SoCoordinate3;
  points->point.setValues(0, 2*num, pts);

  auto *line = new SoIndexedLineSet;
  line->coordIndex.setValues(0, 2*num+4, index);

  auto *face = new SoIndexedFaceSet;
  face->coordIndex.setValues(0, 13*(num-1)+5, indexf);

  soSepRigidBody->addChild(points);
  soSepRigidBody->addChild(face);
  soSepRigidBody->addChild(soOutLineSwitch);
  soOutLineSep->addChild(line);
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
