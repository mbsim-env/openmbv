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
#include "coilspring.h"
#include "mainwindow.h"
#include "utils.h"
#include "openmbvcppinterface/coilspring.h"

using namespace std;

CoilSpring::CoilSpring(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent) : DynamicColoredBody(obj, parentItem, soParent) {
  coilSpring=(OpenMBV::CoilSpring*)obj;
  iconFile=":/coilspring.svg";
  setIcon(0, Utils::QIconCached(iconFile.c_str()));

  //h5 dataset
  int rows=coilSpring->getRows();
  double dt;
  if(rows>=2) dt=coilSpring->getRow(1)[0]-coilSpring->getRow(0)[0]; else dt=0;
  resetAnimRange(rows, dt);

  // read XML
  scaleValue=coilSpring->getScaleFactor();

  // create so
  // body
  mat=new SoMaterial;
  soSep->addChild(mat);
  if(!isnan(staticColor)) setColor(mat, staticColor);
  fromPoint=new SoTranslation;
  soSep->addChild(fromPoint);
  rotation=new SoRotation;
  soSep->addChild(rotation);  
  extrusion=new SoVRMLExtrusion;
  soSep->addChild(extrusion);

  // cross section
  extrusion->crossSection.setNum(iCircSegments+1);
  SbVec2f *cs = extrusion->crossSection.startEditing();
  for(int i=0;i<iCircSegments;i++) cs[i] = SbVec2f(coilSpring->getCrossSectionRadius()*cos(i*2.*M_PI/iCircSegments), -coilSpring->getCrossSectionRadius()*sin(i*2.*M_PI/iCircSegments)); // clockwise in local coordinate system
  cs[iCircSegments]=cs[0]; // close cross section: uses exact the same point: helpfull for "binary space partitioning container"
  extrusion->crossSection.finishEditing();
  extrusion->crossSection.setDefault(FALSE);

  // initialise spine 
  spine = new float[3*(numberOfSpinePoints+1)];
  for(int i=0;i<=numberOfSpinePoints;i++) {
    spine[3*i] = coilSpring->getSpringRadius()*cos(i*coilSpring->getNumberOfCoils()*2.*M_PI/numberOfSpinePoints);
    spine[3*i+1] = coilSpring->getSpringRadius()*sin(i*coilSpring->getNumberOfCoils()*2.*M_PI/numberOfSpinePoints);
    spine[3*i+2] = 0.;
  }
  extrusion->spine.setValuesPointer(numberOfSpinePoints+1,spine);
  extrusion->spine.setDefault(FALSE);

  // additional flags
  extrusion->solid=TRUE; // backface culling
  extrusion->convex=TRUE; // only convex polygons included in visualisation
  extrusion->ccw=TRUE; // vertex ordering counterclockwise?
  extrusion->beginCap=TRUE; // front side at begin of the spine
  extrusion->endCap=TRUE; // front side at end of the spine
  extrusion->creaseAngle=1.5; // angle below which surface normals are drawn smooth (always smooth, except begin/end cap => < 90deg)
}

QString CoilSpring::getInfo() {
  float x, y, z;
  fromPoint->translation.getValue().getValue(x,y,z);
  return DynamicColoredBody::getInfo()+
         QString("<hr width=\"10000\"/>")+
         QString("<b>From Point:</b> %1, %2, %3<br/>").arg(x).arg(y).arg(z)+
         QString("<b>Length:</b> %1").arg(spine[2*numberOfSpinePoints-3+2]);
}

double CoilSpring::update() {
  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()->getValue();
  std::vector<double> data=coilSpring->getRow(frame);

  // translation / rotation
  fromPoint->translation.setValue(data[1],data[2],data[3]);
  SbVec3f distance(data[4]-data[1],data[5]-data[2],data[6]-data[3]);
  rotation->rotation.setValue(SbRotation(SbVec3f(0,0,1),distance));

  // spine 
  for(int i=0;i<=numberOfSpinePoints;i++) {
    spine[3*i+2] = i*distance.length()*scaleValue/numberOfSpinePoints;
  }
  extrusion->spine.touch();
  
  // color
  if(isnan(staticColor)) setColor(mat, data[7]);

  return data[0];
}

