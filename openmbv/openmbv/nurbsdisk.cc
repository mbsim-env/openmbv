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

  // create so
  // material
  SoMaterial *mat=new SoMaterial;
  soSep->addChild(mat);
  mat->shininess.setValue(0.9);
  if(!isnan(staticColor)) setColor(mat, staticColor);

  // body
  controlPts = new SoCoordinate3;
  soSep->addChild(controlPts);

  surface=new SoIndexedNurbsSurface;
  soSep->addChild(surface);

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
  int nurbsLength = (nr+1)*(nj+degAzimuthal);
  float pointData[nurbsLength+nj*drawDegree*4][3]; 
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

  // vector of the position of the disk
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

  controlPts->point.setValues(0, nurbsLength*3+4*3*nj*drawDegree,pointData);

  // surface
  float knotVecRadial_[nr+1+degRadial+1];
  float knotVecAzimuthal_[nj+1+2*degAzimuthal];
  for(int i=0;i<nr+1+degRadial+1;i++) knotVecRadial_[i]=knotVecRadial[i];
  for(int i=0;i<nj+1+2*degAzimuthal;i++) knotVecAzimuthal_[i]=knotVecAzimuthal[i];

  surface->numVControlPoints = nr+1;
  surface->numUControlPoints = nj+degAzimuthal;
  surface->vKnotVector.setValues(0, nr+1+degRadial+1, knotVecRadial_);
  surface->uKnotVector.setValues(0, nj+1+2*degAzimuthal, knotVecAzimuthal_);

  int nurbsIndices[nurbsLength];
  for(int i=0;i<(nr+1);i++) 
    for(int j=0;j<(nj+degAzimuthal);j++) 
      nurbsIndices[i*(nj+degAzimuthal)+j]=(i+1)*(nj+degAzimuthal)-1-j;
  surface->coordIndex.setValues(0,nurbsLength,nurbsIndices);

  // faces for primitive closure
  int faceValues[(nj*drawDegree)*3*5];
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

  faceSet->coordIndex.setValues(0,(nj*drawDegree)*3*5,faceValues);

  return data[0];
}

