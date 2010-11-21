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
#include "utils.h"
#include <Inventor/SoInput.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/actions/SoCallbackAction.h>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include "SoSpecial.h"

using namespace std;

bool Utils::init=false;

void Utils::initialize() {
  if(init==true) return;
  init=true;

  // tess
# ifndef WIN32
#   define __stdcall
# endif
  gluTessCallback(tess, GLU_TESS_BEGIN_DATA, (void (__stdcall *)())tessBeginCB);
  gluTessCallback(tess, GLU_TESS_VERTEX, (void (__stdcall *)())tessVertexCB);
  gluTessCallback(tess, GLU_TESS_END, (void (__stdcall *)())tessEndCB);
}

const QIcon& Utils::QIconCached(const QString& filename) {
  static unordered_map<string, QIcon> myIconCache;
  pair<unordered_map<string, QIcon>::iterator, bool> ins=myIconCache.insert(pair<string, QIcon>(filename.toStdString(), QIcon()));
  if(ins.second)
    return ins.first->second=QIcon(filename);
  return ins.first->second;
}

SoSeparator* Utils::SoDBreadAllCached(const string &filename) {
  static unordered_map<string, SoSeparator*> myIvBodyCache;
  pair<unordered_map<string, SoSeparator*>::iterator, bool> ins=myIvBodyCache.insert(pair<string, SoSeparator*>(filename, (SoSeparator*)NULL));
  if(ins.second) {
    SoInput in;
    in.openFile(filename.c_str());
    return ins.first->second=SoDB::readAll(&in);
  }
  return ins.first->second;
}

// convenience: create frame so
SoSeparator* Utils::soFrame(double size, double offset, bool pickBBoxAble, SoScale *&scale) {
  SoSeparator *sep;
  if(pickBBoxAble)
    sep=new SoSeparator;
  else
    sep=new SoSepNoPickNoBBox;
  sep->ref();

  SoBaseColor *col;
  SoLineSet *line;

  // coordinates
  scale=new SoScale;
  sep->addChild(scale);
  scale->scaleFactor.setValue(size, size, size);
  SoCoordinate3 *coord=new SoCoordinate3;
  sep->addChild(coord);
  coord->point.set1Value(0, -1.0/2+offset*1.0/2, 0, 0);
  coord->point.set1Value(1, +1.0/2+offset*1.0/2, 0, 0);
  coord->point.set1Value(2, 0, -1.0/2+offset*1.0/2, 0);
  coord->point.set1Value(3, 0, +1.0/2+offset*1.0/2, 0);
  coord->point.set1Value(4, 0, 0, -1.0/2+offset*1.0/2);
  coord->point.set1Value(5, 0, 0, +1.0/2+offset*1.0/2);

  // x-axis
  col=new SoBaseColor;
  col->rgb=SbColor(1, 0, 0);
  sep->addChild(col);
  line=new SoLineSet;
  line->startIndex.setValue(0);
  line->numVertices.setValue(2);
  sep->addChild(line);

  // y-axis
  col=new SoBaseColor;
  col->rgb=SbColor(0, 1, 0);
  sep->addChild(col);
  line=new SoLineSet;
  line->startIndex.setValue(2);
  line->numVertices.setValue(2);
  sep->addChild(line);

  // z-axis
  col=new SoBaseColor;
  col->rgb=SbColor(0, 0, 1);
  sep->addChild(col);
  line=new SoLineSet;
  line->startIndex.setValue(4);
  line->numVertices.setValue(2);
  sep->addChild(line);

  return sep;
}

SbRotation Utils::cardan2Rotation(const SbVec3f &c) {
  float a, b, g;
  c.getValue(a,b,g);
  return SbRotation(SbMatrix(
    cos(b)*cos(g),
    -cos(b)*sin(g),
    sin(b),
    0.0,

    cos(a)*sin(g)+sin(a)*sin(b)*cos(g),
    cos(a)*cos(g)-sin(a)*sin(b)*sin(g),
    -sin(a)*cos(b),
    0.0,

    sin(a)*sin(g)-cos(a)*sin(b)*cos(g),
    sin(a)*cos(g)+cos(a)*sin(b)*sin(g),
    cos(a)*cos(b),
    0.0,

    0.0,
    0.0,
    0.0,
    1.0
  ));
}

SbVec3f Utils::rotation2Cardan(const SbRotation& R) {
  SbMatrix M;
  R.getValue(M);
  float a, b, g;    
  b=asin(M[0][2]);
  double nenner=cos(b);
  if (nenner>1e-10) {
    a=atan2(-M[1][2],M[2][2]);
    g=atan2(-M[0][1],M[0][0]);
  } else {
    a=0;
    g=atan2(M[1][0],M[1][1]);
  }
  return SbVec3f(a,b,g);
}

struct CoordEdge { 
  SoCoordinate3 *coord;
  Utils::Edges *edges;
};
SoCoordinate3* Utils::preCalculateEdgesCached(SoGroup *grp, Utils::Edges *&edges) {
  static std::map<SoGroup*, CoordEdge> myCache;
  std::map<SoGroup*, CoordEdge>::iterator i=myCache.find(grp);
  if(i==myCache.end()) {
    CoordEdge xx;
    xx.coord=Utils::preCalculateEdges(grp, edges);
    xx.edges=edges;
    return (myCache[grp]=xx).coord;
  }
  edges=i->second.edges;
  return i->second.coord;
}

void Utils::triangleCB(void *data, SoCallbackAction *action, const SoPrimitiveVertex *vp1, const SoPrimitiveVertex *vp2, const SoPrimitiveVertex *vp3) {
  Utils::Edges* edges=(Utils::Edges*)data;
  // get coordinates of points
  SbVec3f v1=vp1->getPoint();
  SbVec3f v2=vp2->getPoint();
  SbVec3f v3=vp3->getPoint();
  // convert coordinates to frame of the action
  SbMatrix mm=action->getModelMatrix();
  mm.multVecMatrix(v1, v1);
  mm.multVecMatrix(v2, v2);
  mm.multVecMatrix(v3, v3);
  // add point and get point index
  unsigned int v1i=edges->vertex.addPoint(v1);
  unsigned int v2i=edges->vertex.addPoint(v2);
  unsigned int v3i=edges->vertex.addPoint(v3);
  // add edge and get edge index
#define addEdge(i,j) addPoint(SbVec3f(i<j?i:j,i<j?j:i,0))
  unsigned int e1i=edges->edge.addEdge(v1i, v2i);
  unsigned int e2i=edges->edge.addEdge(v2i, v3i);
  unsigned int e3i=edges->edge.addEdge(v3i, v1i);
#undef addEdge
  // add normal
  int ni=edges->normal.getLength();
  SbVec3f n=(v2-v1).cross(v3-v1);
  n.normalize();
  edges->normal.append(&n);
  // add ei->ni,v1i,v2i
  Utils::EI2VINI *xx;
#define expand(ei) if(ei>=edges->ei2vini.size()) edges->ei2vini.resize(ei+1);
  expand(e1i); xx=&edges->ei2vini[e1i]; xx->vai=v1i; xx->vbi=v2i; xx->ni.push_back(ni);
  expand(e2i); xx=&edges->ei2vini[e2i]; xx->vai=v2i; xx->vbi=v3i; xx->ni.push_back(ni);
  expand(e3i); xx=&edges->ei2vini[e3i]; xx->vai=v3i; xx->vbi=v1i; xx->ni.push_back(ni);
#undef expand
}

SoCoordinate3* Utils::preCalculateEdges(SoGroup *sep, Utils::Edges *&edges) {
  if(edges==NULL) edges=new Utils::Edges;
  // get all triangles
  SoCallbackAction cba;
  cba.addTriangleCallback(SoShape::getClassTypeId(), triangleCB, edges);
  cba.apply(sep);
  // return vertex
  SoCoordinate3 *soEdgeVertex=new SoCoordinate3;
  soEdgeVertex->point.setValuesPointer(edges->vertex.numPoints(), edges->vertex.getPointsArrayPtr());
  return soEdgeVertex;
}

SoIndexedLineSet* Utils::calculateCreaseEdges(const double creaseAngle, const Utils::Edges *edges) {
  SoIndexedLineSet *soCreaseEdge=new SoIndexedLineSet;
  int nr=0;
  for(unsigned int i=0; i<edges->ei2vini.size(); i++) {
    if(edges->ei2vini[i].ni.size()==2) {
      int nia=edges->ei2vini[i].ni[0];
      int nib=edges->ei2vini[i].ni[1];
      int vai=edges->ei2vini[i].vai;
      int vbi=edges->ei2vini[i].vbi;
      if(edges->normal[nia]->dot(*edges->normal[nib])<cos(creaseAngle)) {
        soCreaseEdge->coordIndex.set1Value(nr++, vai);
        soCreaseEdge->coordIndex.set1Value(nr++, vbi);
        soCreaseEdge->coordIndex.set1Value(nr++, -1);
      }
    }
  }
  return soCreaseEdge;
}

SoIndexedLineSet* Utils::calculateBoundaryEdges(const Utils::Edges *edges) {
  SoIndexedLineSet *soBoundaryEdge=new SoIndexedLineSet;
  int nr=0;
  for(unsigned int i=0; i<edges->ei2vini.size(); i++) {
    if(edges->ei2vini[i].ni.size()==1) {
      //int ni, v1i, v2i;
      soBoundaryEdge->coordIndex.set1Value(nr++, edges->ei2vini[i].vai);
      soBoundaryEdge->coordIndex.set1Value(nr++, edges->ei2vini[i].vbi);
      soBoundaryEdge->coordIndex.set1Value(nr++, -1);
    }
  }
  return soBoundaryEdge;
}

SoIndexedLineSet* Utils::calculateShilouetteEdge(const SbVec3f &n, const Edges *edges) {
  SoIndexedLineSet *ls=new SoIndexedLineSet;
  int nr=0;
  for(unsigned int i=0; i<edges->ei2vini.size(); i++) {
    if(edges->ei2vini[i].ni.size()==2) {
      int nia=edges->ei2vini[i].ni[0];
      int nib=edges->ei2vini[i].ni[1];
      int vai=edges->ei2vini[i].vai;
      int vbi=edges->ei2vini[i].vbi;
      if(edges->normal[nia]->dot(n)*edges->normal[nib]->dot(n)<=0) {
        ls->coordIndex.set1Value(nr++, vai);
        ls->coordIndex.set1Value(nr++, vbi);
        ls->coordIndex.set1Value(nr++, -1);
      }
    }
  }
  return ls;
}

// for tess
GLUtesselator *Utils::tess=gluNewTess();
GLenum Utils::tessType;
int Utils::tessNumVertices;
SoTriangleStripSet *Utils::tessTriangleStrip;
SoIndexedFaceSet *Utils::tessTriangleFan;
SoCoordinate3 *Utils::tessCoord;

// tess
void Utils::tessBeginCB(GLenum type, void *data) {
  SoGroup *parent=(SoGroup*)data;
  tessType=type;
  tessNumVertices=0;
  tessCoord=new SoCoordinate3;
  parent->addChild(tessCoord);
  if(tessType==GL_TRIANGLES || tessType==GL_TRIANGLE_STRIP) {
    tessTriangleStrip=new SoTriangleStripSet;
    parent->addChild(tessTriangleStrip);
  }
  if(tessType==GL_TRIANGLE_FAN) {
    tessTriangleFan=new SoIndexedFaceSet;
    parent->addChild(tessTriangleFan);
  }
}

void Utils::tessVertexCB(GLdouble *vertex) {
  tessCoord->point.set1Value(tessNumVertices++, vertex[0], vertex[1], vertex[2]);
}

void Utils::tessEndCB(void) {
  if(tessType==GL_TRIANGLES)
    for(int i=0; i<tessNumVertices/3; i++)
      tessTriangleStrip->numVertices.set1Value(i, 3);
  if(tessType==GL_TRIANGLE_STRIP)
    tessTriangleStrip->numVertices.set1Value(0, tessNumVertices);
  if(tessType==GL_TRIANGLE_FAN) {
    int j=0;
    for(int i=0; i<tessNumVertices-2; i++) {
      tessTriangleFan->coordIndex.set1Value(j++, 0);
      tessTriangleFan->coordIndex.set1Value(j++, i+1);
      tessTriangleFan->coordIndex.set1Value(j++, i+2);
      tessTriangleFan->coordIndex.set1Value(j++, -1);
    }
  }
}
