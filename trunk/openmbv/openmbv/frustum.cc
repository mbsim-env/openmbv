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
#include "frustum.h"
#include <vector>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoIndexedTriangleStripSet.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/nodes/SoIndexedLineSet.h>
#include <Inventor/nodes/SoNormal.h>
#include <Inventor/nodes/SoShapeHints.h>
#include "utils.h"
#include "openmbvcppinterface/frustum.h"
#include <QMenu>
#include <cfloat>

using namespace std;

Frustum::Frustum(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  OpenMBV::Frustum *f=(OpenMBV::Frustum*)obj;
  iconFile=":/frustum.svg";
  setIcon(0, Utils::QIconCached(iconFile.c_str()));

  // read XML
  double baseRadius=f->getBaseRadius();
  double topRadius=f->getTopRadius();
  double height=f->getHeight();
  if(fabs(height)<1e-13) height=0;
  double innerBaseRadius=f->getInnerBaseRadius();
  double innerTopRadius=f->getInnerTopRadius();

  const int N=20;

  // create so
  // two side render if height==0
  if(height==0) {
    SoShapeHints *sh=new SoShapeHints;
    soSepRigidBody->addChild(sh);
    sh->vertexOrdering.setValue(SoShapeHints::CLOCKWISE);
  }
  // coordinates
  SoCoordinate3 *coord=new SoCoordinate3;
  soSepRigidBody->addChild(coord);
  for(int i=0; i<N; i++) {
    double phi=2*M_PI/N*i;
    coord->point.set1Value(i+0, baseRadius*cos(phi), baseRadius*sin(phi), -height);
    coord->point.set1Value(i+N, topRadius*cos(phi), topRadius*sin(phi), 0);
    if(innerBaseRadius>0 || innerTopRadius>0) {
      coord->point.set1Value(i+2*N, innerBaseRadius*cos(phi), innerBaseRadius*sin(phi), -height);
      coord->point.set1Value(i+3*N, innerTopRadius*cos(phi), innerTopRadius*sin(phi), 0);
    }
  }
  // normals
  SoNormal *normal=new SoNormal;
  soSepRigidBody->addChild(normal);
  normal->vector.set1Value(0, 0, 0, -1);
  normal->vector.set1Value(1, 0, 0, 1);
  for(int i=0; i<N; i++) {
    double phi=2*M_PI/N*i;
    normal->vector.set1Value(i+2, cos(phi), sin(phi), (baseRadius-topRadius)/height);
    if(innerBaseRadius>0 || innerTopRadius>0)
      normal->vector.set1Value(i+2+N, -cos(phi), -sin(phi), -(innerBaseRadius-innerTopRadius)/height);
  }
  // fix radius if height==0
  if(height==0 && innerTopRadius<innerBaseRadius) innerBaseRadius=innerTopRadius;
  if(height==0 && topRadius>baseRadius) baseRadius=topRadius;
  // faces (base/top)
  if(innerBaseRadius>0 || innerTopRadius>0) {
    int nr=-1;
    SoIndexedTriangleStripSet *baseFace=new SoIndexedTriangleStripSet;
    soSepRigidBody->addChild(baseFace);
    SoIndexedTriangleStripSet *topFace=0;
    if(height!=0) {
      topFace=new SoIndexedTriangleStripSet;
      soSepRigidBody->addChild(topFace);
    }
    baseFace->coordIndex.set1Value(++nr, N-1);
    baseFace->normalIndex.set1Value(nr, 0);
    if(height!=0) {
      topFace->coordIndex.set1Value(nr, 2*N-1);
      topFace->normalIndex.set1Value(nr, 1);
    }
    baseFace->coordIndex.set1Value(++nr, 3*N-1);
    baseFace->normalIndex.set1Value(nr, 0);
    if(height!=0) {
      topFace->coordIndex.set1Value(nr, 4*N-1);
      topFace->normalIndex.set1Value(nr, 1);
    }
    for(int i=0; i<N; i++) {
      baseFace->coordIndex.set1Value(++nr, i);
      baseFace->normalIndex.set1Value(nr, 0);
      if(height!=0) {
        topFace->coordIndex.set1Value(nr, i+N);
        topFace->normalIndex.set1Value(nr, 1);
      }
      baseFace->coordIndex.set1Value(++nr, i+2*N);
      baseFace->normalIndex.set1Value(nr, 0);
      if(height!=0) {
        topFace->coordIndex.set1Value(nr, i+3*N);
        topFace->normalIndex.set1Value(nr, 1);
      }
    }
  }
  else {
    int nr=-1;
    SoIndexedFaceSet *baseFace=new SoIndexedFaceSet;
    soSepRigidBody->addChild(baseFace);
    SoIndexedFaceSet *topFace=0;
    if(height!=0) {
      topFace=new SoIndexedFaceSet;
      soSepRigidBody->addChild(topFace);
    }
    for(int i=0; i<N; i++) {
      baseFace->coordIndex.set1Value(++nr, i);
      baseFace->normalIndex.set1Value(nr, 0);
      if(height!=0) {
        topFace->coordIndex.set1Value(nr, i+N);
        topFace->normalIndex.set1Value(nr, 1);
      }
    }
  }
  if(height!=0) {
    // faces outer
    int nr=-1;
    SoIndexedTriangleStripSet *outerFace=new SoIndexedTriangleStripSet;
    soSepRigidBody->addChild(outerFace);
    outerFace->coordIndex.set1Value(++nr, N-1);
    outerFace->normalIndex.set1Value(nr, N-1+2);
    outerFace->coordIndex.set1Value(++nr, 2*N-1);
    outerFace->normalIndex.set1Value(nr, N-1+2);
    for(int i=0; i<N; i++) {
      outerFace->coordIndex.set1Value(++nr, i);
      outerFace->normalIndex.set1Value(nr, i+2);
      outerFace->coordIndex.set1Value(++nr, i+N);
      outerFace->normalIndex.set1Value(nr, i+2);
    }
    // faces inner
    if(innerBaseRadius>0 || innerTopRadius>0) {
      int nr=-1;
      SoIndexedTriangleStripSet *innerFace=new SoIndexedTriangleStripSet;
      soSepRigidBody->addChild(innerFace);
      innerFace->coordIndex.set1Value(++nr, 3*N-1);
      innerFace->normalIndex.set1Value(nr, 2*N-1+2);
      innerFace->coordIndex.set1Value(++nr, 4*N-1);
      innerFace->normalIndex.set1Value(nr, 2*N-1+2);
      for(int i=0; i<N; i++) {
        innerFace->coordIndex.set1Value(++nr, 2*N+i);
        innerFace->normalIndex.set1Value(nr, i+2+N);
        innerFace->coordIndex.set1Value(++nr, i+3*N);
        innerFace->normalIndex.set1Value(nr, i+2+N);
      }
    }
  }
  // scale ref/localFrame
  double size=min(2*max(baseRadius,topRadius),height)*f->getScaleFactor();
  refFrameScale->scaleFactor.setValue(size,size,size);
  localFrameScale->scaleFactor.setValue(size,size,size);
  
  // outline
  SoIndexedLineSet *outLine=new SoIndexedLineSet;
  int nr=-1;
  outLine->coordIndex.set1Value(++nr, N-1);
  for(int i=0; i<N; i++)
    outLine->coordIndex.set1Value(++nr, i);
  outLine->coordIndex.set1Value(++nr, -1);
  if(height!=0) {
    outLine->coordIndex.set1Value(++nr, 2*N-1);
    for(int i=0; i<N; i++)
      outLine->coordIndex.set1Value(++nr, i+N);
  }
  if(innerBaseRadius>0 || innerTopRadius>0) {
    outLine->coordIndex.set1Value(++nr, -1);
    outLine->coordIndex.set1Value(++nr, 3*N-1);
    for(int i=0; i<N; i++)
      outLine->coordIndex.set1Value(++nr, i+2*N);
    outLine->coordIndex.set1Value(++nr, -1);
    if(height!=0) {
      outLine->coordIndex.set1Value(++nr, 4*N-1);
      for(int i=0; i<N; i++)
        outLine->coordIndex.set1Value(++nr, i+3*N);
    }
  }
  soSepRigidBody->addChild(soOutLineSwitch);
  soOutLineSep->addChild(outLine);

  // GUI editors
  if(!clone) {
    properties->updateHeader();
    FloatEditor *baseRadiusEditor=new FloatEditor(properties, QIcon(), "Base radius");
    baseRadiusEditor->setRange(0, DBL_MAX);
    baseRadiusEditor->setOpenMBVParameter(f, &OpenMBV::Frustum::getBaseRadius, &OpenMBV::Frustum::setBaseRadius);

    FloatEditor *topRadiusEditor=new FloatEditor(properties, QIcon(), "Top radius");
    topRadiusEditor->setRange(0, DBL_MAX);
    topRadiusEditor->setOpenMBVParameter(f, &OpenMBV::Frustum::getTopRadius, &OpenMBV::Frustum::setTopRadius);

    FloatEditor *heightEditor=new FloatEditor(properties, QIcon(), "Height");
    heightEditor->setRange(0, DBL_MAX);
    heightEditor->setOpenMBVParameter(f, &OpenMBV::Frustum::getHeight, &OpenMBV::Frustum::setHeight);

    FloatEditor *innerBaseRadiusEditor=new FloatEditor(properties, QIcon(), "Inner base radius");
    innerBaseRadiusEditor->setRange(0, DBL_MAX);
    innerBaseRadiusEditor->setOpenMBVParameter(f, &OpenMBV::Frustum::getInnerBaseRadius, &OpenMBV::Frustum::setInnerBaseRadius);

    FloatEditor *innerTopRadiusEditor=new FloatEditor(properties, QIcon(), "Inner top radius");
    innerTopRadiusEditor->setRange(0, DBL_MAX);
    innerTopRadiusEditor->setOpenMBVParameter(f, &OpenMBV::Frustum::getInnerTopRadius, &OpenMBV::Frustum::setInnerTopRadius);
  }
}
