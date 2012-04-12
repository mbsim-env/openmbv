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
#include <Inventor/nodes/SoLineSet.h>

using namespace std;

CoilSpring::CoilSpring(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : DynamicColoredBody(obj, parentItem, soParent, ind) {
  coilSpring=(OpenMBV::CoilSpring*)obj;
  iconFile=":/coilspring.svg";
  setIcon(0, Utils::QIconCached(iconFile.c_str()));

  //h5 dataset
  int rows=coilSpring->getRows();
  double dt;
  if(rows>=2) dt=coilSpring->getRow(1)[0]-coilSpring->getRow(0)[0]; else dt=0;
  resetAnimRange(rows, dt);

  double R=coilSpring->getSpringRadius();
  double r=coilSpring->getCrossSectionRadius();
  N=coilSpring->getNumberOfCoils();
  if(r<0) {
    if(coilSpring->getType()==OpenMBV::CoilSpring::polyline)
      r=2;
    else
      r=R/7;
  }

  // read XML
  scaleValue=coilSpring->getScaleFactor();
  nominalLength=coilSpring->getNominalLength();
  if(nominalLength<0) nominalLength=r*N*4;

  // create so
  // body
  mat=new SoMaterial;
  soSep->addChild(mat);
  if(!isnan(staticColor)) setColor(mat, staticColor);
  fromPoint=new SoTranslation;
  soSep->addChild(fromPoint);
  rotation=new SoRotation;
  soSep->addChild(rotation);  
  soSwitch=new SoSwitch;
  soSep->addChild(soSwitch);
  // tube
  extrusion=new SoVRMLExtrusion;
  soSwitch->addChild(extrusion);
  // scaledTube
  SoSeparator *scaledTubeSep=new SoSeparator;
  soSwitch->addChild(scaledTubeSep);
  scale=new SoScale;
  scaledTubeSep->addChild(scale);
  SoVRMLExtrusion *scaledExtrusion=new SoVRMLExtrusion;
  scaledTubeSep->addChild(scaledExtrusion);
  // polyline
  SoSeparator *polylineSep=new SoSeparator;
  soSwitch->addChild(polylineSep);
  polylineSep->addChild(scale);
  SoDrawStyle *ds=new SoDrawStyle;
  polylineSep->addChild(ds);
  ds->lineWidth.setValue(r);
  SoCoordinate3 *polylineCoord=new SoCoordinate3;
  polylineSep->addChild(polylineCoord);
  SoLineSet *polyline=new SoLineSet;
  polylineSep->addChild(polyline);
  polyline->numVertices.setValue(int(numberOfSpinePointsPerCoil*N));

  // type
  OpenMBV::CoilSpring::Type type=coilSpring->getType();
  if(type==OpenMBV::CoilSpring::tube) soSwitch->whichChild.setValue(0);
  if(type==OpenMBV::CoilSpring::scaledTube) soSwitch->whichChild.setValue(1);
  if(type==OpenMBV::CoilSpring::polyline) soSwitch->whichChild.setValue(2);

  // cross section
  extrusion->crossSection.setNum(iCircSegments+1);
  scaledExtrusion->crossSection.setNum(iCircSegments+1);
  SbVec2f *cs = extrusion->crossSection.startEditing();
  SbVec2f *scs = scaledExtrusion->crossSection.startEditing();
  for(int i=0;i<iCircSegments;i++) // clockwise in local coordinate system
    scs[i]=cs[i]=SbVec2f(r*cos(i*2.*M_PI/iCircSegments), -r*sin(i*2.*M_PI/iCircSegments));
  scs[iCircSegments]=cs[iCircSegments]=cs[0]; // close cross section: uses exact the same point: helpfull for "binary space partitioning container"
  extrusion->crossSection.finishEditing();
  scaledExtrusion->crossSection.finishEditing();
  extrusion->crossSection.setDefault(FALSE);
  scaledExtrusion->crossSection.setDefault(FALSE);

  // initialise spine 
  spine = new float[3*(int(numberOfSpinePointsPerCoil*N)+1)];
  scaledSpine = new float[3*(int(numberOfSpinePointsPerCoil*N)+1)];
  for(int i=0;i<=numberOfSpinePointsPerCoil*N;i++) {
    scaledSpine[3*i]=spine[3*i] = R*cos(i*N*2.*M_PI/numberOfSpinePointsPerCoil/N);
    scaledSpine[3*i+1]=spine[3*i+1] = R*sin(i*N*2.*M_PI/numberOfSpinePointsPerCoil/N);
    spine[3*i+2] = 0;
    scaledSpine[3*i+2] = i*nominalLength/numberOfSpinePointsPerCoil/N;
  }
  extrusion->spine.setValuesPointer(int(numberOfSpinePointsPerCoil*N+1),spine);
  scaledExtrusion->spine.setValuesPointer(int(numberOfSpinePointsPerCoil*N+1),scaledSpine);
  polylineCoord->point.setValuesPointer(int(numberOfSpinePointsPerCoil*N+1),scaledSpine);
  extrusion->spine.setDefault(FALSE);
  scaledExtrusion->spine.setDefault(FALSE);

  // additional flags
  extrusion->solid=TRUE; // backface culling
  scaledExtrusion->solid=TRUE; // backface culling
  extrusion->convex=TRUE; // only convex polygons included in visualisation
  scaledExtrusion->convex=TRUE; // only convex polygons included in visualisation
  extrusion->ccw=TRUE; // vertex ordering counterclockwise?
  scaledExtrusion->ccw=TRUE; // vertex ordering counterclockwise?
  extrusion->beginCap=TRUE; // front side at begin of the spine
  scaledExtrusion->beginCap=TRUE; // front side at begin of the spine
  extrusion->endCap=TRUE; // front side at end of the spine
  scaledExtrusion->endCap=TRUE; // front side at end of the spine
  extrusion->creaseAngle=1.5; // angle below which surface normals are drawn smooth (always smooth, except begin/end cap => < 90deg)
  scaledExtrusion->creaseAngle=1.5; // angle below which surface normals are drawn smooth (always smooth, except begin/end cap => < 90deg)

  // gui
  typeAct=new QActionGroup(this);
  typeTube=new QAction("Tube", typeAct);
  typeScaledTube=new QAction("Scaled tube", typeAct);
  typePolyline=new QAction("Polyline", typeAct);
  typeTube->setCheckable(true);
  typeTube->setData(QVariant(OpenMBV::CoilSpring::tube));
  typeTube->setObjectName("CoilSpring::typeTube");
  typeScaledTube->setCheckable(true);
  typeScaledTube->setData(QVariant(OpenMBV::CoilSpring::scaledTube));
  typeScaledTube->setObjectName("CoilSpring::typeScaledTube");
  typePolyline->setCheckable(true);
  typePolyline->setData(QVariant(OpenMBV::CoilSpring::polyline));
  typePolyline->setObjectName("CoilSpring::typePolyline");
  switch(coilSpring->getType()) {
    case OpenMBV::CoilSpring::tube: typeTube->setChecked(true); break;
    case OpenMBV::CoilSpring::scaledTube: typeScaledTube->setChecked(true); break;
    case OpenMBV::CoilSpring::polyline: typePolyline->setChecked(true); break;
  }
  connect(typeAct,SIGNAL(triggered(QAction*)),this,SLOT(typeSlot(QAction*)));
}

CoilSpring::~CoilSpring() {
  delete[]spine;
  delete[]scaledSpine;
}

QString CoilSpring::getInfo() {
  float x, y, z;
  fromPoint->translation.getValue().getValue(x,y,z);
  float sx, sy, sz=0;
  if(soSwitch->whichChild.getValue()!=0)
    scale->scaleFactor.getValue().getValue(sx, sy, sz);
  return DynamicColoredBody::getInfo()+
         QString("<hr width=\"10000\"/>")+
         QString("<b>From point:</b> %1, %2, %3<br/>").arg(x).arg(y).arg(z)+
         QString("<b>Length:</b> %1").arg(soSwitch->whichChild.getValue()==0?
                                          spine[3*int(numberOfSpinePointsPerCoil*N)+2]:
                                          sz*nominalLength);
}

double CoilSpring::update() {
  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()->getValue();
  std::vector<double> data=coilSpring->getRow(frame);

  // translation / rotation
  fromPoint->translation.setValue(data[1],data[2],data[3]);
  SbVec3f distance(data[4]-data[1],data[5]-data[2],data[6]-data[3]);
  rotation->rotation.setValue(SbRotation(SbVec3f(0,0,1),distance));

  switch(soSwitch->whichChild.getValue()) {
    case 0:
      // tube 
      for(int i=0;i<=numberOfSpinePointsPerCoil*N;i++) {
        spine[3*i+2] = i*distance.length()*scaleValue/numberOfSpinePointsPerCoil/N;
      }
      extrusion->spine.touch();
      break;
    case 1:
    case 2:
      scale->scaleFactor.setValue(1,1,distance.length()*scaleValue/nominalLength);
      break;
  }
  
  // color
  if(isnan(staticColor)) setColor(mat, data[7]);

  return data[0];
}

QMenu* CoilSpring::createMenu() {
  QMenu* menu=DynamicColoredBody::createMenu();
  menu->addSeparator()->setText("Properties from: CoilSpring");
  menu->addAction(typeTube);
  menu->addAction(typeScaledTube);
  menu->addAction(typePolyline);
  return menu;
}

void CoilSpring::typeSlot(QAction *action) {
  OpenMBV::CoilSpring::Type type=(OpenMBV::CoilSpring::Type)action->data().toInt();
  if(type==OpenMBV::CoilSpring::tube) {
    soSwitch->whichChild.setValue(0);
    coilSpring->setType(OpenMBV::CoilSpring::tube);
  }
  else if(type==OpenMBV::CoilSpring::scaledTube) {
    soSwitch->whichChild.setValue(1);
    coilSpring->setType(OpenMBV::CoilSpring::scaledTube);
  }
  else {
    soSwitch->whichChild.setValue(2);
    coilSpring->setType(OpenMBV::CoilSpring::polyline);
  }
  update();
}
