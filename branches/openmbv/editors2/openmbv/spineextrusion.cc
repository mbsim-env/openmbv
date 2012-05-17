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
#include "spineextrusion.h"
#include "mainwindow.h"
#include <Inventor/nodes/SoMaterial.h>
#include "utils.h"
#include "openmbvcppinterface/spineextrusion.h"
#include <cfloat>

using namespace std;

SpineExtrusion::SpineExtrusion(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : DynamicColoredBody(obj, parentItem, soParent, ind), numberOfSpinePoints(0) {
  spineExtrusion=(OpenMBV::SpineExtrusion*)obj;
  //h5 dataset
  numberOfSpinePoints = int((spineExtrusion->getRow(1).size()-1)/4);
  int rows=spineExtrusion->getRows();
  double dt;
  if(rows>=2) dt=spineExtrusion->getRow(1)[0]-spineExtrusion->getRow(0)[0]; else dt=0;
  resetAnimRange(rows, dt);

  // read XML
  vector<OpenMBV::PolygonPoint*>* contour=spineExtrusion->getContour();

  // create so
  // material
  SoMaterial *mat=new SoMaterial;
  soSep->addChild(mat);
  mat->shininess.setValue(0.9);
  if(!isnan(staticColor)) setColor(mat, staticColor);

  // body
  extrusion=new SoVRMLExtrusion;
  soSep->addChild(extrusion);
  twistAxis = new SbVec3f(0.,1.,0.);

  // scale
  extrusion->scale.setNum(numberOfSpinePoints);
  SbVec2f *sc = extrusion->scale.startEditing();
  for(int i=0;i<numberOfSpinePoints;i++) sc[i] = SbVec2f(spineExtrusion->getScaleFactor(),spineExtrusion->getScaleFactor()); // first x-scale / second z-scale
  extrusion->scale.finishEditing();
  extrusion->scale.setDefault(FALSE);

  // cross section
  extrusion->crossSection.setNum(contour?contour->size()+1:0);
  SbVec2f *cs = extrusion->crossSection.startEditing();
  for(size_t i=0;i<(contour?contour->size():0);i++) cs[i] = SbVec2f((*contour)[i]->getXComponent(), (*contour)[i]->getYComponent()); // clockwise in local coordinate system
  if(contour) cs[contour->size()] =  SbVec2f((*contour)[0]->getXComponent(), (*contour)[0]->getYComponent()); // closed cross section
  extrusion->crossSection.finishEditing();
  extrusion->crossSection.setDefault(FALSE);

  // additional flags
  extrusion->solid=TRUE; // backface culling
  extrusion->convex=TRUE; // only convex polygons included in visualisation
  extrusion->ccw=TRUE; // vertex ordering counterclockwise?
  extrusion->beginCap=TRUE; // front side at begin of the spine
  extrusion->endCap=TRUE; // front side at end of the spine
  extrusion->creaseAngle=0.3; // angle below which surface normals are drawn smooth

  if(!clone) {
    properties->updateHeader();
    // GUI editors
    FloatMatrixEditor *contourEditor=new FloatMatrixEditor(properties, QIcon(), "Contour", 0, 3);
    contourEditor->setOpenMBVParameter(spineExtrusion, &OpenMBV::SpineExtrusion::getContour, &OpenMBV::SpineExtrusion::setContour);

    FloatEditor *scaleFactorEditor=new FloatEditor(properties, QIcon(), "Scale factor");
    scaleFactorEditor->setRange(0, DBL_MAX);
    scaleFactorEditor->setOpenMBVParameter(spineExtrusion, &OpenMBV::SpineExtrusion::getScaleFactor, &OpenMBV::SpineExtrusion::setScaleFactor);
  }
}

QString SpineExtrusion::getInfo() {
  return DynamicColoredBody::getInfo();
}

double SpineExtrusion::update() {
  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()->getValue();
  std::vector<double> data=spineExtrusion->getRow(frame);

  // set spine
  extrusion->spine.setNum(numberOfSpinePoints);
  SbVec3f *sp = extrusion->spine.startEditing();
  for(int i=0;i<numberOfSpinePoints;i++) {
    sp[i] = SbVec3f(data[4*i+1],data[4*i+2],data[4*i+3]);
  }
  extrusion->spine.finishEditing();
  extrusion->spine.setDefault(FALSE);

  // set twist
  extrusion->orientation.setNum(numberOfSpinePoints);
  SbRotation *tw = extrusion->orientation.startEditing();
  for(int i=0;i<numberOfSpinePoints;i++) {
    tw[i] = SbRotation(*twistAxis,data[4*i+4]);
  }
  extrusion->orientation.finishEditing();
  extrusion->orientation.setDefault(FALSE);

  return data[0];
}

