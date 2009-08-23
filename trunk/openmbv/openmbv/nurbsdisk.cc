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
  TiXmlElement *e=element->FirstChildElement(OPENMBVNS"scaleFactor");
  double scaleValue=toVector(e->GetText())[0];
  e=element->FirstChildElement(OPENMBVNS"drawDegree");
  drawDegree=(int)toVector(e->GetText())[0];  
  e=element->FirstChildElement(OPENMBVNS"elementNumberAzimuthal");
  nj=(int)toVector(e->GetText())[0];
  e=element->FirstChildElement(OPENMBVNS"elementNumberRadial");
  nr=(int)toVector(e->GetText())[0]; 
  e=element->FirstChildElement(OPENMBVNS"interpolationDegreeRadial");
  degRadial=(int)toVector(e->GetText())[0];
  e=element->FirstChildElement(OPENMBVNS"interpolationDegreeAzimuthal");
  degAzimuthal=(int)toVector(e->GetText())[0];
  e=element->FirstChildElement(OPENMBVNS"innerRadius");
  innerRadius=toVector(e->GetText())[0];
  e=element->FirstChildElement(OPENMBVNS"outerRadius");
  outerRadius=toVector(e->GetText())[0];
  e=element->FirstChildElement(OPENMBVNS"knotVecAzimuthal");
  knotVecAzimuthal=toVector(e->GetText());
  e=element->FirstChildElement(OPENMBVNS"knotVecRadial");
  knotVecRadial=toVector(e->GetText());  

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
  soSep->addChild(faceSet);  
}

QString NurbsDisk::getInfo() {
  return DynamicColoredBody::getInfo();
}

double NurbsDisk::update() {
  if(h5Group==0) return 0;
  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()->getValue();
  std::vector<double> data=h5Data->getRow(frame);

  // set points
  controlPts->point.setNum(nurbsLength+nj*drawDegree*4);
  SbVec3f *pointData = controlPts->point.startEditing();
  for(int i=0;i<nurbsLength+nj*drawDegree;i++) {  // control points + visualisation points of the inner ring
    pointData[i][0]=data[i*3+1];
    pointData[i][1]=data[i*3+2];
    pointData[i][2]=data[i*3+3];
  }

  for(int i=nurbsLength+nj*drawDegree;i<nurbsLength+2*nj*drawDegree;i++) { // visualisation points of the outer ring
    pointData[i+2*nj*drawDegree][0]=data[i*3+1];
    pointData[i+2*nj*drawDegree][1]=data[i*3+2];
    pointData[i+2*nj*drawDegree][2]=data[i*3+3];
  }

  // vector of the position of the disk (midpoint of base circle, not midplane!)
  float DiskPosition[3];
  for(int i=0;i<3;i++) DiskPosition[i]=data[1+nurbsLength*3+2*nj*drawDegree*3+i];

  // rotary matrix
  float Orientation[3][3];
  for(int i=0;i<3;i++) 
    for(int j=0;j<3;j++) 
      Orientation[i][j]=data[1+nurbsLength*3+2*nj*drawDegree*3+3+i*3+j];

  // point for sides and back of the disk for visualisation
  float pointtmp[3], point[3];
  for(int i=0; i<nj*drawDegree;i++) {
    float Phi=2*M_PI*i/((nj)*drawDegree);
    
    //inner Ring
    point[0]=cos(Phi) * innerRadius;
    point[1]=sin(Phi) * innerRadius;
    point[2]=0.;
    for(int j=0;j<3;j++) pointtmp[j]=Orientation[j][0]*point[0]+Orientation[j][1]*point[1]+Orientation[j][2]*point[2]; 
    pointData[nurbsLength+(nj)*1*drawDegree+i][0]=pointtmp[0]+DiskPosition[0];
    pointData[nurbsLength+(nj)*1*drawDegree+i][1]=pointtmp[1]+DiskPosition[1];
    pointData[nurbsLength+(nj)*1*drawDegree+i][2]=pointtmp[2]+DiskPosition[2];

    // outer Ring
    point[0]=cos(Phi) * outerRadius;
    point[1]=sin(Phi) * outerRadius;
    point[2]=0.;
    for(int j=0;j<3;j++) pointtmp[j]=Orientation[j][0]*point[0]+Orientation[j][1]*point[1]+Orientation[j][2]*point[2];
    pointData[nurbsLength+(nj)*2*drawDegree+i][0]=pointtmp[0]+DiskPosition[0];
    pointData[nurbsLength+(nj)*2*drawDegree+i][1]=pointtmp[1]+DiskPosition[1];
    pointData[nurbsLength+(nj)*2*drawDegree+i][2]=pointtmp[2]+DiskPosition[2];
  }
  controlPts->point.finishEditing();
  controlPts->point.setDefault(FALSE);

  // faces
  faceSet->coordIndex.setNum((nj*drawDegree)*3*5);
  int32_t *faceValues = faceSet->coordIndex.startEditing();
  for(int j=0;j<3;j++) { // inner ring up-down ; circle - hollow down ; outer down-up  
    for(int i=0;i<(nj*drawDegree-1);i++) { 
      faceValues[j*nj*drawDegree*5+i*5+0]=nurbsLength+(nj*drawDegree)*(j)+i;
      faceValues[j*nj*drawDegree*5+i*5+1]=nurbsLength+(nj*drawDegree)*(j)+i+1;
      faceValues[j*nj*drawDegree*5+i*5+2]=nurbsLength+(nj*drawDegree)*(j+1)+i+1;
      faceValues[j*nj*drawDegree*5+i*5+3]=nurbsLength+(nj*drawDegree)*(j+1)+i;
      faceValues[j*nj*drawDegree*5+i*5+4]=-1;
    }

    // changeover from last point to first point
    faceValues[j*nj*drawDegree*5+(nj*drawDegree-1)*5+0]=nurbsLength+(nj*drawDegree)*(j)+(nj*drawDegree-1);
    faceValues[j*nj*drawDegree*5+(nj*drawDegree-1)*5+1]=nurbsLength+(nj*drawDegree)*(j);
    faceValues[j*nj*drawDegree*5+(nj*drawDegree-1)*5+2]=nurbsLength+(nj*drawDegree)*(j+1);
    faceValues[j*nj*drawDegree*5+(nj*drawDegree-1)*5+3]=nurbsLength+(nj*drawDegree)*(j+1)+(nj*drawDegree-1);
    faceValues[j*nj*drawDegree*5+(nj*drawDegree-1)*5+4]=-1;
  }
  faceSet->coordIndex.finishEditing();
  faceSet->coordIndex.setDefault(FALSE);
  return data[0];
}

