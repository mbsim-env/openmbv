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

using namespace std;

CoilSpring::CoilSpring(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : Body(element, h5Parent, parentItem, soParent) {
  iconFile=":/coilspring.svg";
  setIcon(0, QIcon(iconFile.c_str()));

  //h5 dataset
  h5Data=new H5::VectorSerie<double>;
  if(h5Group) {
    h5Data->open(*h5Group, "data");
    int rows=h5Data->getRows();
    double dt;
    if(rows>=2) dt=h5Data->getRow(1)[0]-h5Data->getRow(0)[0]; else dt=0;
    resetAnimRange(rows, dt);
  }

  // read XML
  TiXmlElement *e=element->FirstChildElement(OPENMBVNS"numberOfCoils");
  numberOfCoils=(int)toVector(e->GetText())[0];
  e=element->FirstChildElement(OPENMBVNS"springRadius");
  springRadius=toVector(e->GetText())[0];
  e=element->FirstChildElement(OPENMBVNS"crossSectionRadius");
  double crossSectionRadius=toVector(e->GetText())[0];
  e=element->FirstChildElement(OPENMBVNS"scaleFactor");
  scaleValue=toVector(e->GetText())[0];

  // create so
  // body
  mat=new SoMaterial;
  soSep->addChild(mat);
  fromPoint=new SoTranslation;
  soSep->addChild(fromPoint);
  rotation=new SoRotation;
  soSep->addChild(rotation);  
  extrusion=new SoVRMLExtrusion;
  soSep->addChild(extrusion);

  // cross section
  extrusion->crossSection.setNum(iCircSegments+1);
  SbVec2f *cs = extrusion->crossSection.startEditing();
  for(int i=0;i<=iCircSegments;i++) cs[i] = SbVec2f(crossSectionRadius*cos(i*2.*M_PI/iCircSegments), -crossSectionRadius*sin(i*2.*M_PI/iCircSegments)); // clockwise in local coordinate system
  extrusion->crossSection.finishEditing();
  extrusion->crossSection.setDefault(FALSE);

  // initialise spine 
  spine = new float[3*(numberOfSpinePoints+1)];
  for(int i=0;i<=numberOfSpinePoints;i++) {
    spine[3*i] = springRadius*cos(i*numberOfCoils*2.*M_PI/numberOfSpinePoints);
    spine[3*i+1] = springRadius*sin(i*numberOfCoils*2.*M_PI/numberOfSpinePoints);
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
  extrusion->creaseAngle=0.3; // angle below which surface normals are drawn smooth
}

QString CoilSpring::getInfo() {
  float x, y, z;
  fromPoint->translation.getValue().getValue(x,y,z);
  return Body::getInfo()+
         QString("-----<br/>")+
         QString("<b>From Point:</b> %1, %2, %3<br/>").arg(x).arg(y).arg(z)+
         QString("<b>Length:</b> %1<br/>").arg(spine[2*numberOfSpinePoints-3+2]);
}

double CoilSpring::update() {
  if(h5Group==0) return 0;
  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()->getValue();
  std::vector<double> data=h5Data->getRow(frame);

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
  mat->diffuseColor.setHSVValue((1-data[7])*2/3,1,1);
  mat->specularColor.setHSVValue((1-data[7])*2/3,0.7,1);

  return data[0];
}

