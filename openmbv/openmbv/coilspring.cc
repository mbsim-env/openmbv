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
#include "coilspring.h"
#include "mainwindow.h"
#include "utils.h"
#include "openmbvcppinterface/coilspring.h"
#include <Inventor/nodes/SoLineSet.h>
#include <QMenu>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

CoilSpring::CoilSpring(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : DynamicColoredBody(obj, parentItem, soParent, ind), spine(nullptr), scaledSpine(nullptr) {
  coilSpring=std::static_pointer_cast<OpenMBV::CoilSpring>(obj);
  iconFile="coilspring.svg";
  setIcon(0, Utils::QIconCached(iconFile));

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
  fromPoint=new SoTranslation;
  soSep->addChild(fromPoint);
  rotation=new SoRotation;
  soSep->addChild(rotation);  

  switch(coilSpring->getType()) {
    case OpenMBV::CoilSpring::tube: {
      extrusion=new SoVRMLExtrusion;
      soSep->addChild(extrusion);
      // cross section
      extrusion->crossSection.setNum(iCircSegments+1);
      SbVec2f *cs = extrusion->crossSection.startEditing();
      for(int i=0;i<iCircSegments;i++) // clockwise in local coordinate system
        cs[i]=SbVec2f(r*cos(i*2.*M_PI/iCircSegments), -r*sin(i*2.*M_PI/iCircSegments));
      cs[iCircSegments]=cs[0]; // close cross section: uses exact the same point: helpfull for "binary space partitioning container"
      extrusion->crossSection.finishEditing();
      extrusion->crossSection.setDefault(FALSE);
      // initialise spine 
      spine = new float[3*(int(numberOfSpinePointsPerCoil*N)+1)];
      for(int i=0;i<=numberOfSpinePointsPerCoil*N;i++) {
        spine[3*i] = R*cos(i*N*2.*M_PI/numberOfSpinePointsPerCoil/N);
        spine[3*i+1] = R*sin(i*N*2.*M_PI/numberOfSpinePointsPerCoil/N);
        spine[3*i+2] = 0;
      }
      extrusion->spine.setValuesPointer(int(numberOfSpinePointsPerCoil*N+1),spine);
      extrusion->spine.setDefault(FALSE);
      // additional flags
      extrusion->solid=TRUE; // backface culling
      extrusion->convex=TRUE; // only convex polygons included in visualisation
      extrusion->ccw=TRUE; // vertex ordering counterclockwise?
      extrusion->beginCap=TRUE; // front side at begin of the spine
      extrusion->endCap=TRUE; // front side at end of the spine
      extrusion->creaseAngle=1.5; // angle below which surface normals are drawn smooth (always smooth, except begin/end cap => < 90deg)
      break;
    }
    case OpenMBV::CoilSpring::scaledTube: {
      auto *scaledTubeSep=new SoSeparator;
      soSep->addChild(scaledTubeSep);
      scale=new SoScale;
      scaledTubeSep->addChild(scale);
      auto *scaledExtrusion=new SoVRMLExtrusion;
      scaledTubeSep->addChild(scaledExtrusion);
      // cross section
      scaledExtrusion->crossSection.setNum(iCircSegments+1);
      SbVec2f *scs = scaledExtrusion->crossSection.startEditing();
      for(int i=0;i<iCircSegments;i++) // clockwise in local coordinate system
        scs[i]=SbVec2f(r*cos(i*2.*M_PI/iCircSegments), -r*sin(i*2.*M_PI/iCircSegments));
      scs[iCircSegments]=scs[0]; // close cross section: uses exact the same point: helpfull for "binary space partitioning container"
      scaledExtrusion->crossSection.finishEditing();
      scaledExtrusion->crossSection.setDefault(FALSE);
      // initialise spine 
      scaledSpine = new float[3*(int(numberOfSpinePointsPerCoil*N)+1)];
      for(int i=0;i<=numberOfSpinePointsPerCoil*N;i++) {
        scaledSpine[3*i]= R*cos(i*N*2.*M_PI/numberOfSpinePointsPerCoil/N);
        scaledSpine[3*i+1]= R*sin(i*N*2.*M_PI/numberOfSpinePointsPerCoil/N);
        scaledSpine[3*i+2] = i*nominalLength/numberOfSpinePointsPerCoil/N;
      }
      scaledExtrusion->spine.setValuesPointer(int(numberOfSpinePointsPerCoil*N+1),scaledSpine);
      scaledExtrusion->spine.setDefault(FALSE);
      // additional flags
      scaledExtrusion->solid=TRUE; // backface culling
      scaledExtrusion->convex=TRUE; // only convex polygons included in visualisation
      scaledExtrusion->ccw=TRUE; // vertex ordering counterclockwise?
      scaledExtrusion->beginCap=TRUE; // front side at begin of the spine
      scaledExtrusion->endCap=TRUE; // front side at end of the spine
      scaledExtrusion->creaseAngle=1.5; // angle below which surface normals are drawn smooth (always smooth, except begin/end cap => < 90deg)
      break;
    }
    case OpenMBV::CoilSpring::polyline: {
      auto *polylineSep=new SoSeparator;
      soSep->addChild(polylineSep);
      scale=new SoScale;
      polylineSep->addChild(scale);
      auto *ds=new SoDrawStyle;
      polylineSep->addChild(ds);
      ds->lineWidth.setValue(r);
      auto *polylineCoord=new SoCoordinate3;
      polylineSep->addChild(polylineCoord);
      auto *polyline=new SoLineSet;
      polylineSep->addChild(polyline);
      polyline->numVertices.setValue(int(numberOfSpinePointsPerCoil*N));
      // initialise spine 
      scaledSpine = new float[3*(int(numberOfSpinePointsPerCoil*N)+1)];
      for(int i=0;i<=numberOfSpinePointsPerCoil*N;i++) {
        scaledSpine[3*i]= R*cos(i*N*2.*M_PI/numberOfSpinePointsPerCoil/N);
        scaledSpine[3*i+1]= R*sin(i*N*2.*M_PI/numberOfSpinePointsPerCoil/N);
        scaledSpine[3*i+2] = i*nominalLength/numberOfSpinePointsPerCoil/N;
      }
      polylineCoord->point.setValuesPointer(int(numberOfSpinePointsPerCoil*N+1),scaledSpine);
      break;
    }
  }
}

void CoilSpring::createProperties() {
  DynamicColoredBody::createProperties();

  // GUI editors
  if(!clone) {
    properties->updateHeader();
    auto *typeEditor=new ComboBoxEditor(properties, QIcon(), "Type", {
      make_tuple(OpenMBV::CoilSpring::tube,       "Tube",        QIcon(), "CoilSpring::type::tube"),
      make_tuple(OpenMBV::CoilSpring::scaledTube, "Scaled tube", QIcon(), "CoilSpring::type::scaledTube"),
      make_tuple(OpenMBV::CoilSpring::polyline,   "Polyline",    QIcon(), "CoilSpring::type::polyline")
    });
    typeEditor->setOpenMBVParameter(coilSpring, &OpenMBV::CoilSpring::getType, &OpenMBV::CoilSpring::setType);
    properties->addPropertyActionGroup(typeEditor->getActionGroup());

    auto *numberOfCoilsEditor=new FloatEditor(properties, QIcon(), "Number of coils");
    numberOfCoilsEditor->setRange(0, DBL_MAX);
    numberOfCoilsEditor->setOpenMBVParameter(coilSpring, &OpenMBV::CoilSpring::getNumberOfCoils, &OpenMBV::CoilSpring::setNumberOfCoils);

    auto *springRadiusEditor=new FloatEditor(properties, QIcon(), "Coil spring radius");
    springRadiusEditor->setRange(0, DBL_MAX);
    springRadiusEditor->setOpenMBVParameter(coilSpring, &OpenMBV::CoilSpring::getSpringRadius, &OpenMBV::CoilSpring::setSpringRadius);

    auto *crossSectionRadiusEditor=new FloatEditor(properties, QIcon(), "Cross section radius");
    crossSectionRadiusEditor->setRange(0, DBL_MAX);
    crossSectionRadiusEditor->setOpenMBVParameter(coilSpring, &OpenMBV::CoilSpring::getCrossSectionRadius, &OpenMBV::CoilSpring::setCrossSectionRadius);

    auto *nominalLengthEditor=new FloatEditor(properties, QIcon(), "Nominal length");
    nominalLengthEditor->setRange(0, DBL_MAX);
    nominalLengthEditor->setOpenMBVParameter(coilSpring, &OpenMBV::CoilSpring::getNominalLength, &OpenMBV::CoilSpring::setNominalLength);

    auto *scaleFactorEditor=new FloatEditor(properties, QIcon(), "Scale factor");
    scaleFactorEditor->setOpenMBVParameter(coilSpring, &OpenMBV::CoilSpring::getScaleFactor, &OpenMBV::CoilSpring::setScaleFactor);
  }
}

CoilSpring::~CoilSpring() {
  delete[]spine;
  delete[]scaledSpine;
}

QString CoilSpring::getInfo() {
  float x, y, z;
  fromPoint->translation.getValue().getValue(x,y,z);
  float sx, sy, sz=0;
  if(coilSpring->getType()!=OpenMBV::CoilSpring::tube)
    scale->scaleFactor.getValue().getValue(sx, sy, sz);
  return DynamicColoredBody::getInfo()+
         QString("<hr width=\"10000\"/>")+
         QString("<b>From point:</b> %1, %2, %3<br/>").arg(x).arg(y).arg(z)+
         QString("<b>Length:</b> %1").arg(coilSpring->getType()==OpenMBV::CoilSpring::tube?
                                          spine[3*int(numberOfSpinePointsPerCoil*N)+2]:
                                          sz*nominalLength);
}

double CoilSpring::update() {
  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()->getValue();
  auto data=coilSpring->getRow(frame);

  // translation / rotation
  fromPoint->translation.setValue(data[1],data[2],data[3]);
  SbVec3f distance(data[4]-data[1],data[5]-data[2],data[6]-data[3]);
  rotation->rotation.setValue(SbRotation(SbVec3f(0,0,1),distance));

  switch(coilSpring->getType()) {
    case OpenMBV::CoilSpring::tube:
      // tube 
      for(int i=0;i<=numberOfSpinePointsPerCoil*N;i++) {
        spine[3*i+2] = i*distance.length()*scaleValue/numberOfSpinePointsPerCoil/N;
      }
      extrusion->spine.touch();
      break;
    case OpenMBV::CoilSpring::scaledTube:
    case OpenMBV::CoilSpring::polyline:
      scale->scaleFactor.setValue(1,1,distance.length()*scaleValue/nominalLength);
      break;
  }
  
  // color
  if(diffuseColor[0]<0) setColor(data[7]);

  return data[0];
}

}
