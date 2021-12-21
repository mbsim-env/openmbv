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
  int indl[2*num+4];
  int k=0;
  for(unsigned int i=0; i<num; i++)
    indl[k++] = i;
  indl[k++] = 0;
  indl[k++] = -1;
  for(unsigned int i=num; i<2*num; i++)
    indl[k++] = i;
  indl[k++] = num;
  indl[k++] = -1;

  int indf[13*(num-1)+5];
  k=0;
  for(unsigned int i=0; i<num-1; i++) {
    indf[k++] = 0;
    indf[k++] = i;
    indf[k++] = i+1;
    indf[k++] = -1;
  }
  for(unsigned int i=num; i<2*num-1; i++) {
    indf[k++] = num;
    indf[k++] = i+1;
    indf[k++] = i;
    indf[k++] = -1;
  }
  for(unsigned int i=0; i<num-1; i++) {
    indf[k++] = num+i;
    indf[k++] = num+i+1;
    indf[k++] = i+1;
    indf[k++] = i;
    indf[k++] = -1;
  }
  indf[k++] = 2*num-1;
  indf[k++] = num;
  indf[k++] = 0;
  indf[k++] = num-1;
  indf[k++] = -1;

  auto *points = new SoCoordinate3;
  points->point.setValues(0, 2*num, pts);

  auto *line = new SoIndexedLineSet;
  line->coordIndex.setValues(0, 2*num+4, indl);

  auto *face = new SoIndexedFaceSet;
  face->coordIndex.setValues(0, 13*(num-1)+5, indf);

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
