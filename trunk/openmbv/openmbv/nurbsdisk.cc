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
#include "nurbsdisk.h"
#include "mainwindow.h"
#include <Inventor/nodes/SoMaterial.h>

using namespace std;

NurbsDisk::NurbsDisk(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : DynamicColoredBody(element, h5Parent, parentItem, soParent) {
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
  TiXmlElement *e;
  e=element->FirstChildElement(OPENMBVNS"scaleFactor");

  // create so
  // material
  SoMaterial *mat=new SoMaterial;
  soSep->addChild(mat);
  mat->shininess.setValue(0.9);
  if(!isnan(staticColor)) setColor(mat, staticColor);

  // body
//  extrusion=new SoVRMLExtrusion;
//  soSep->addChild(extrusion);
//  twistAxis = new SbVec3f(0.,1.,0.);
//
//  // scale
//  extrusion->scale.setNum(numberOfSpinePoints);
//  SbVec2f *sc = extrusion->scale.startEditing();
//  for(int i=0;i<numberOfSpinePoints;i++) sc[i] = SbVec2f(scaleValue,scaleValue); // first x-scale / second z-scale
//  extrusion->scale.finishEditing();
//  extrusion->scale.setDefault(FALSE);
//
//  // cross section
//  extrusion->crossSection.setNum(contour.size()+1);
//  SbVec2f *cs = extrusion->crossSection.startEditing();
//  for(int i=0;i<contour.size();i++) cs[i] = SbVec2f(contour[i][0], contour[i][1]); // clockwise in local coordinate system
//  cs[contour.size()] =  SbVec2f(contour[0][0], contour[0][1]); // closed cross section
//  extrusion->crossSection.finishEditing();
//  extrusion->crossSection.setDefault(FALSE);
//
//  // additional flags
//  extrusion->solid=TRUE; // backface culling
//  extrusion->convex=TRUE; // only convex polygons included in visualisation
//  extrusion->ccw=TRUE; // vertex ordering counterclockwise?
//  extrusion->beginCap=FALSE; // front side at begin of the spine
//  extrusion->endCap=FALSE; // front side at end of the spine
//  extrusion->creaseAngle=0.3; // angle below which surface normals are drawn smooth
}

QString NurbsDisk::getInfo() {
  return DynamicColoredBody::getInfo();
}

double NurbsDisk::update() {
  if(h5Group==0) return 0;
  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()->getValue();
  std::vector<double> data=h5Data->getRow(frame);

//  // set spine
//  extrusion->spine.setNum(numberOfSpinePoints);
//  SbVec3f *sp = extrusion->spine.startEditing();
//  for(int i=0;i<numberOfSpinePoints;i++) {
//    sp[i] = SbVec3f(data[4*i+1],data[4*i+2],data[4*i+3]);
//  }
//  extrusion->spine.finishEditing();
//  extrusion->spine.setDefault(FALSE);
//
//  // set twist
//  extrusion->orientation.setNum(numberOfSpinePoints);
//  SbRotation *tw = extrusion->orientation.startEditing();
//  for(int i=0;i<numberOfSpinePoints;i++) {
//    tw[i] = SbRotation(*twistAxis,data[4*i+4]);
//  }
//  extrusion->orientation.finishEditing();
//  extrusion->orientation.setDefault(FALSE);

  return data[0];
}

