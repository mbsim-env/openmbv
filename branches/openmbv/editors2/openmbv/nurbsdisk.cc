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
#include "utils.h"
#include "openmbvcppinterface/nurbsdisk.h"
#include <cfloat>

using namespace std;

NurbsDisk::NurbsDisk(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : DynamicColoredBody(obj, parentItem, soParent, ind) {
  nurbsDisk=(OpenMBV::NurbsDisk*)obj;
  //h5 dataset
  int rows=nurbsDisk->getRows();
  double dt;
  if(rows>=2) dt=nurbsDisk->getRow(1)[0]-nurbsDisk->getRow(0)[0]; else dt=0;
  resetAnimRange(rows, dt);

  // read XML
  drawDegree=(int)(nurbsDisk->getDrawDegree());
  nj=nurbsDisk->getElementNumberAzimuthal();
  nr=nurbsDisk->getElementNumberRadial();
  degRadial=nurbsDisk->getInterpolationDegreeRadial();
  degAzimuthal=nurbsDisk->getInterpolationDegreeAzimuthal();
  innerRadius=nurbsDisk->getRi();
  outerRadius=nurbsDisk->getRo();
  vector<double> dummy;
  dummy=nurbsDisk->getKnotVecAzimuthal();
  knotVecAzimuthal.clear();
  for(int i=0; i<nj+1+2*degAzimuthal; i++)
    knotVecAzimuthal.push_back(dummy[i]);
  dummy=nurbsDisk->getKnotVecRadial();
  knotVecRadial.clear();
  for(int i=0; i<nr+2+degRadial+1; i++)
    knotVecRadial.push_back(dummy[i]);

  nurbsLength = (nr+1)*(nj+degAzimuthal);

  // create so
  // material
  SoMaterial *mat=new SoMaterial;
  soSep->addChild(mat);
  mat->shininess.setValue(0.9);
  if(!isnan(staticColor)) setColor(mat, staticColor);

  // body
  // points
  controlPts = new SoCoordinate3;
  soSep->addChild(controlPts);

  // nurbs
  surface=new SoIndexedNurbsSurface;
  soSep->addChild(surface);

  //surface->numVControlPoints = 4;
  //surface->numUControlPoints = 4;

  surface->numVControlPoints = nr+1;
  surface->numUControlPoints = nj+degAzimuthal;

  // IN COMMENTS: BEZIER SURFACE FOR TEST
  //surface->vKnotVector.setNum(8);
  //float* knotVecRadial_ = surface->vKnotVector.startEditing();
  //for(int i=0;i<4;i++) knotVecRadial_[i]=0.;
  //for(int i=4;i<8;i++) knotVecRadial_[i]=1.;
  //surface->vKnotVector.finishEditing();
  //surface->vKnotVector.setDefault(FALSE);

  surface->vKnotVector.setNum(nr+1+degRadial+1);
  float* knotVecRadial_ = surface->vKnotVector.startEditing();
  for(int i=0;i<nr+1+degRadial+1;i++)
    knotVecRadial_[i]=knotVecRadial[i];
  surface->vKnotVector.finishEditing();
  surface->vKnotVector.setDefault(FALSE);

  //surface->uKnotVector.setNum(8);
  //float* knotVecAzimuthal_ = surface->uKnotVector.startEditing();
  //for(int i=0;i<4;i++) knotVecAzimuthal_[i]=0.;
  //for(int i=4;i<8;i++) knotVecAzimuthal_[i]=1.;
  //surface->uKnotVector.finishEditing();
  //surface->uKnotVector.setDefault(FALSE);

  surface->uKnotVector.setNum(nj+1+2*degAzimuthal);
  float* knotVecAzimuthal_ = surface->uKnotVector.startEditing();
  for(int i=0;i<nj+1+2*degAzimuthal;i++)
    knotVecAzimuthal_[i]=knotVecAzimuthal[i];
  surface->uKnotVector.finishEditing();
  surface->uKnotVector.setDefault(FALSE);

  //surface->coordIndex.setNum(16);
  //int32_t *nurbsIndices = surface->coordIndex.startEditing();
  //for(int i=0;i<4;i++)
  //  for(int j=0;j<4;j++)
  //    nurbsIndices[i*4+j]=i*4+j;
  //surface->coordIndex.finishEditing();
  //surface->coordIndex.setDefault(FALSE);

  surface->coordIndex.setNum(nurbsLength);
  int32_t *nurbsIndices = surface->coordIndex.startEditing();
  for(int i=0;i<(nr+1);i++)
    for(int j=0;j<(nj+degAzimuthal);j++)
      nurbsIndices[i*(nj+degAzimuthal)+j]=(i+1)*(nj+degAzimuthal)-1-j;
  surface->coordIndex.finishEditing();
  surface->coordIndex.setDefault(FALSE);

  //controlPts->point.setNum(16);
  //SbVec3f *pointData = controlPts->point.startEditing();
  //pointData[0][0]  =-4.5; pointData[0][1]  =-2.0; pointData[0][2]  = 8.0;
  //pointData[1][0]  =-2.0; pointData[1][1]  = 1.0; pointData[1][2]  = 8.0;
  //pointData[2][0]  = 2.0; pointData[2][1]  =-3.0; pointData[2][2]  = 6.0;
  //pointData[3][0]  = 5.0; pointData[3][1]  =-1.0; pointData[3][2]  = 8.0;
  //pointData[4][0]  =-3.0; pointData[4][1]  = 3.0; pointData[4][2]  = 4.0;
  //pointData[5][0]  = 0.0; pointData[5][1]  =-1.0; pointData[5][2]  = 4.0;
  //pointData[6][0]  = 1.0; pointData[6][1]  =-1.0; pointData[6][2]  = 4.0;
  //pointData[7][0]  = 3.0; pointData[7][1]  = 2.0; pointData[7][2]  = 4.0;
  //pointData[8][0]  =-5.0; pointData[8][1]  =-2.0; pointData[8][2]  =-2.0;
  //pointData[9][0]  =-2.0; pointData[9][1]  =-4.0; pointData[9][2]  =-2.0;
  //pointData[10][0] = 2.0; pointData[10][1] =-1.0; pointData[10][2] =-2.0;
  //pointData[11][0] = 5.0; pointData[11][1] = 0.0; pointData[11][2] =-2.0;
  //pointData[12][0] =-4.5; pointData[12][1] = 2.0; pointData[12][2] =-6.0;
  //pointData[13][0] =-2.0; pointData[13][1] =-4.0; pointData[13][2] =-5.0;
  //pointData[14][0] = 2.0; pointData[14][1] = 3.0; pointData[14][2] =-5.0;
  //pointData[15][0] = 4.5; pointData[15][1] =-2.0; pointData[15][2] =-6.0;
  //controlPts->point.finishEditing();
  //controlPts->point.setDefault(FALSE);

  // faces
  faceSet=new SoIndexedFaceSet;
  faceSet->ref();
  soSep->addChild(faceSet);

  // translation (from hdf5)
  translation=new SoTranslation;
  soSep->addChild(translation);

  // rotation (from hdf5)
  rotationAlpha=new SoRotationXYZ;
  rotationAlpha->axis=SoRotationXYZ::X;
  soSep->addChild(rotationAlpha);
  rotationBeta=new SoRotationXYZ;
  rotationBeta->axis=SoRotationXYZ::Y;
  soSep->addChild(rotationBeta);
  rotationGamma=new SoRotationXYZ;
  rotationGamma->axis=SoRotationXYZ::Z;
  soSep->addChild(rotationGamma);
  rotation=new SoRotation;
  rotation->ref(); // do not add to scene graph (only for convenience)

  // till now the scene graph should be static (except color). So add a SoSeparator for caching
  soSepNurbsDisk=new SoSeparator;
  soSep->addChild(soSepNurbsDisk);

  // reference frame
  soLocalFrameSwitch=new SoSwitch;
  soSepNurbsDisk->addChild(soLocalFrameSwitch);
  soLocalFrameSwitch->addChild(Utils::soFrame(1,1,false,localFrameScale));
  localFrameScale->ref();
  soLocalFrameSwitch->whichChild.setValue(nurbsDisk->getLocalFrame()?SO_SWITCH_ALL:SO_SWITCH_NONE);

  //Add option to move camera with body
  moveCameraWith=new QAction(Utils::QIconCached(":/camerabody.svg"),"Move camera with this body",this);
  moveCameraWith->setObjectName("RigidBody::moveCameraWith");
  connect(moveCameraWith,SIGNAL(triggered()),this,SLOT(moveCameraWithSlot()));

#if 0 
  // GUI editors
  if(!clone) {
    BoolEditor *localFrameEditor=new BoolEditor(properties, Utils::QIconCached(":/localframe.svg"), "Draw local frame");
    localFrameEditor->setOpenMBVParameter(nurbsDisk, &OpenMBV::NurbsDisk::getLocalFrame, &OpenMBV::NurbsDisk::setLocalFrame);
    
    FloatEditor *scaleFactorEditor=new FloatEditor(properties, QIcon(), "Scale factor");
    scaleFactorEditor->setRange(0, DBL_MAX);
    scaleFactorEditor->setOpenMBVParameter(nurbsDisk, &OpenMBV::NurbsDisk::getScaleFactor, &OpenMBV::NurbsDisk::setScaleFactor);

    IntEditor *drawDegreeEditor=new IntEditor(properties, QIcon(), "Draw discretisation");
    drawDegreeEditor->setRange(0, INT_MAX);
    drawDegreeEditor->setOpenMBVParameter(nurbsDisk, &OpenMBV::NurbsDisk::getDrawDegree, &OpenMBV::NurbsDisk::setDrawDegree);

    FloatEditor *innerRadiusEditor=new FloatEditor(properties, QIcon(), "Inner radius");
    innerRadiusEditor->setRange(0, DBL_MAX);
    innerRadiusEditor->setOpenMBVParameter(nurbsDisk, &OpenMBV::NurbsDisk::getRi, &OpenMBV::NurbsDisk::setRi);

    FloatEditor *outerRadiusEditor=new FloatEditor(properties, QIcon(), "Outer radius");
    outerRadiusEditor->setRange(0, DBL_MAX);
    outerRadiusEditor->setOpenMBVParameter(nurbsDisk, &OpenMBV::NurbsDisk::getRo, &OpenMBV::NurbsDisk::setRo);

    IntEditor *elementNumberAzimuthalEditor=new IntEditor(properties, QIcon(), "Number of azimuthal elements");
    elementNumberAzimuthalEditor->setRange(0, INT_MAX);
    elementNumberAzimuthalEditor->setOpenMBVParameter(nurbsDisk, &OpenMBV::NurbsDisk::getElementNumberAzimuthal, &OpenMBV::NurbsDisk::setElementNumberAzimuthal);

    IntEditor *elementNumberRadialEditor=new IntEditor(properties, QIcon(), "Number of radial elements");
    elementNumberRadialEditor->setRange(0, INT_MAX);
    elementNumberRadialEditor->setOpenMBVParameter(nurbsDisk, &OpenMBV::NurbsDisk::getElementNumberRadial, &OpenMBV::NurbsDisk::setElementNumberRadial);

    IntEditor *interpolationDegreeAzimuthalEditor=new IntEditor(properties, QIcon(), "Azimuthal spline degree");
    interpolationDegreeAzimuthalEditor->setRange(0, INT_MAX);
    interpolationDegreeAzimuthalEditor->setOpenMBVParameter(nurbsDisk, &OpenMBV::NurbsDisk::getInterpolationDegreeAzimuthal, &OpenMBV::NurbsDisk::setInterpolationDegreeAzimuthal);

    IntEditor *interpolationDegreeRadialEditor=new IntEditor(properties, QIcon(), "Radial spline degree");
    interpolationDegreeRadialEditor->setRange(0, INT_MAX);
    interpolationDegreeRadialEditor->setOpenMBVParameter(nurbsDisk, &OpenMBV::NurbsDisk::getInterpolationDegreeRadial, &OpenMBV::NurbsDisk::setInterpolationDegreeRadial);

    FloatMatrixEditor *knotVecAzimuthalEditor=new FloatMatrixEditor(properties, QIcon(), "Azimuthal knot vector", 1, 0);
    knotVecAzimuthalEditor->setOpenMBVParameter(nurbsDisk, &OpenMBV::NurbsDisk::getKnotVecAzimuthal, &OpenMBV::NurbsDisk::setKnotVecAzimuthal);

    FloatMatrixEditor *knotVecRadialEditor=new FloatMatrixEditor(properties, QIcon(), "Radial knot vector", 1, 0);
    knotVecRadialEditor->setOpenMBVParameter(nurbsDisk, &OpenMBV::NurbsDisk::getKnotVecRadial, &OpenMBV::NurbsDisk::setKnotVecRadial);
  }
#endif
}

NurbsDisk::~NurbsDisk() {
  faceSet->unref();
  localFrameScale->unref();
  rotation->unref();
}

QString NurbsDisk::getInfo() {
  return DynamicColoredBody::getInfo();
}

void NurbsDisk::moveCameraWithSlot() {
  MainWindow::getInstance()->moveCameraWith(&translation->translation, &rotation->rotation);
}

QMenu* NurbsDisk::createMenu() {
  QMenu* menu=DynamicColoredBody::createMenu();
  menu->addSeparator()->setText("Properties from: NurbsDisk");
  menu->addAction(moveCameraWith);
  return menu;
}

double NurbsDisk::update() {
  // read from hdf5
  int frame = MainWindow::getInstance()->getFrame()->getValue();
  std::vector<double> data = nurbsDisk->getRow(frame);

  // vector of the position of the disk (midpoint of base circle, not midplane!)
  translation->translation.setValue(data[1], data[2], data[3]);

  // set scene values
  rotationAlpha->angle.setValue(data[4]);
  rotationBeta->angle.setValue(data[5]);
  rotationGamma->angle.setValue(data[6]);
  rotation->rotation.setValue(Utils::cardan2Rotation(SbVec3f(data[4], data[5], data[6])).inverse()); // set rotation matrix (needed for move camera with body)

  // rotary matrix
  SbMatrix Orientation;
  rotation->rotation.getValue().getValue(Orientation);
  Orientation = Orientation.transpose();

  // read and set control points
  //number of control points:
  // - nurbsLength for every control point
  // - "nj*drawDegree" for every point on the inner and outer ring (factor 2) on the surface and and the midplane (factor 2) --> factor 4
  controlPts->point.setNum(nurbsLength + 4 * nj*drawDegree);

  SbVec3f *pointData = controlPts->point.startEditing();

  //control points
  for (int i = 0; i < nurbsLength; i++) {
    pointData[i][0] = data[7 + i * 3 + 0];
    pointData[i][1] = data[7 + i * 3 + 1];
    pointData[i][2] = data[7 + i * 3 + 2];
  }

  //inner ring

  //points at the surface (inner ring)
  for(int i = nurbsLength; i < nurbsLength + nj*drawDegree; i++) {
    pointData[i][0] = data[7 + i * 3 + 0];
    pointData[i][1] = data[7 + i * 3 + 1];
    pointData[i][2] = data[7 + i * 3 + 2];
  }

  //points at the midplane (inner ring)
  for(int i = nurbsLength + nj*drawDegree; i < nurbsLength + 2* nj*drawDegree; i++) {
    //point on the midplane in local frame
    double Phi = 2 * M_PI * (i -nurbsLength + nj*drawDegree)  / ((nj) * drawDegree);
    double point[3];

    point[0] = cos(Phi) * innerRadius;
    point[1] = sin(Phi) * innerRadius;
    point[2] = 0.;

    //transform to world frame
    double pointtmp[3];
    for (int j = 0; j < 3; j++)
      pointtmp[j] = Orientation[j][0] * point[0] + Orientation[j][1] * point[1] + Orientation[j][2] * point[2];

    pointData[i][0] = pointtmp[0] + data[1];
    pointData[i][1] = pointtmp[1] + data[2];
    pointData[i][2] = pointtmp[2] + data[3];
  }

  //outer ring

  //points at the surface (outer ring)
  for(int i = nurbsLength + 2 * nj*drawDegree; i < nurbsLength + 3 * nj * drawDegree; i++) {
    int index = i - nj*drawDegree;
    pointData[i][0] = data[7 + index * 3 + 0];
    pointData[i][1] = data[7 + index * 3 + 1];
    pointData[i][2] = data[7 + index * 3 + 2];
  }

  //points at the midplane (inner ring)
  for(int i = nurbsLength + 3 * nj*drawDegree; i < nurbsLength + 4 * nj*drawDegree; i++) {

    //point on the midplane in local frame
    double Phi = 2 * M_PI * (i - nurbsLength + 3 * nj*drawDegree) / ((nj) * drawDegree);
    double point[3];

    point[0] = cos(Phi) * outerRadius;
    point[1] = sin(Phi) * outerRadius;
    point[2] = 0.;


    //transform to world frame
    double pointtmp[3];
    for (int j = 0; j < 3; j++)
      pointtmp[j] = Orientation[j][0] * point[0] + Orientation[j][1] * point[1] + Orientation[j][2] * point[2];

    pointData[i][0] = pointtmp[0] + data[1];
    pointData[i][1] = pointtmp[1] + data[2];
    pointData[i][2] = pointtmp[2] + data[3];
  }

  controlPts->point.finishEditing();
  controlPts->point.setDefault(FALSE);


  //indexing the face set
  vector<int> num;
  num.push_back(5 * nj * drawDegree); //number of points for the midplane
  num.push_back(5 * nj * drawDegree); //number of face sets on the inner ring
  num.push_back(5 * nj * drawDegree); //number of face sets on the outer ring
  faceSet->coordIndex.setNum(num[0] + num[1] + num[2]);

  //5 entries in build one quad (4 points and one "-1" entry to show that the face is finished
  //Remark: The order of the indices decides which side is colored and which side is black / dark
  int32_t *faceIndices = faceSet->coordIndex.startEditing();

  //midplane
  for(int i=0; i < num[0]; i+=5) {
    int index = i/5;
    if(index < nj*drawDegree-1) {
      faceIndices[i] = nurbsLength + 3 * nj*drawDegree + index; //outer ring
      faceIndices[i+1] =   nurbsLength + nj*drawDegree + index; //inner ring
      faceIndices[i+2] = nurbsLength + nj*drawDegree + 1 +  index; //inner ring
      faceIndices[i+3] = nurbsLength + 3 * nj*drawDegree + 1 + index ; //outer ring
    }
    else {//ringclosure
      faceIndices[i]   = nurbsLength + 3 * nj*drawDegree + index; //outer ring
      faceIndices[i+1] = nurbsLength + nj*drawDegree + index; //inner ring
      faceIndices[i+2] = nurbsLength + nj*drawDegree; //inner ring
      faceIndices[i+3] = nurbsLength + 3 * nj*drawDegree; //outer ring

    }

    faceIndices[i+4] = -1;
  }


  //inner ring
  for(int i = num[0]; i < num[0]+num[1]; i+=5) {
    int index = (i-num[0])/5;
    if(index < nj*drawDegree-1) {
      faceIndices[i] = nurbsLength + nj*drawDegree + index; //midplane
      faceIndices[i+1] = nurbsLength + index; //surface
      faceIndices[i+2] = nurbsLength + 1 + index ; //surface
      faceIndices[i+3] = nurbsLength + nj*drawDegree + 1 +  index; //midplane
    }
    else {//ringclosure
      faceIndices[i] = nurbsLength + nj*drawDegree + index; //midplane
      faceIndices[i+1] = nurbsLength + index; //surface
      faceIndices[i+2] = nurbsLength; //surface
      faceIndices[i+3] = nurbsLength + nj*drawDegree; //midplane
    }

    faceIndices[i+4] = -1;
  }

  //outer ring
  for(int i = num[0]+num[1]; i < num[0]+num[1]+num[2]; i+=5) {
    int index = (i-num[0]-num[1])/5;
    if(index < nj*drawDegree-1) {
      faceIndices[i]   = nurbsLength + 2 * nj*drawDegree +  index; //surface
      faceIndices[i+1] = nurbsLength + 3 * nj*drawDegree + index; //midplane
      faceIndices[i+2] = nurbsLength + 3 * nj*drawDegree + 1 +  index; //midplane
      faceIndices[i+3] = nurbsLength + 2 * nj*drawDegree + 1 + index ; //surface

    }
    else {//ringclosure
      faceIndices[i]   = nurbsLength + 2 * nj*drawDegree + index; //surface
      faceIndices[i+1] = nurbsLength + 3 * nj*drawDegree + index; //midplane
      faceIndices[i+2] = nurbsLength + 3 * nj*drawDegree; //midplane
      faceIndices[i+3] = nurbsLength + 2 * nj*drawDegree; //surface
    }

    faceIndices[i+4] = -1;
  }


  faceSet->coordIndex.finishEditing();
  faceSet->coordIndex.setDefault(FALSE);

  return data[0]; //return time
}

