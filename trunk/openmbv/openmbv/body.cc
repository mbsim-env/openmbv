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
#include "body.h"
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoTriangleStripSet.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/nodes/SoRotationXYZ.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoLightModel.h>
#include "SoSpecial.h"
#include <QtGui/QMenu>
#include "mainwindow.h"
#include "compoundrigidbody.h"
#include <GL/gl.h>
#include <Inventor/actions/SoCallbackAction.h>
#include <Inventor/SoPrimitiveVertex.h>

using namespace std;

bool Body::existFiles=false;
Body *Body::timeUpdater=0;

// for tess
GLUtesselator *Body::tess=gluNewTess();
GLenum Body::tessType;
int Body::tessNumVertices;
bool Body::tessCBInit=false;
SoTriangleStripSet *Body::tessTriangleStrip;
SoIndexedFaceSet *Body::tessTriangleFan;
SoCoordinate3 *Body::tessCoord;

Body::Body(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : Object(element, h5Parent, parentItem, soParent) {
  // tess
  if(tessCBInit==false) {
#   ifndef WIN32
#     define __stdcall
#   endif
    gluTessCallback(tess, GLU_TESS_BEGIN_DATA, (void (__stdcall *)())tessBeginCB);
    gluTessCallback(tess, GLU_TESS_VERTEX, (void (__stdcall *)())tessVertexCB);
    gluTessCallback(tess, GLU_TESS_END, (void (__stdcall *)())tessEndCB);
    tessCBInit=true;
  }

  if(h5Parent) {
    // read XML
    TiXmlElement *e=element->FirstChildElement(OPENMBVNS"hdf5Link");
    if(e); // hdf5Link

    // register callback function on frame change
    SoFieldSensor *sensor=new SoFieldSensor(frameSensorCB, this);
    sensor->attach(MainWindow::getInstance()->getFrame());
    sensor->setPriority(0); // is needed for png export
  }

  // switch for outline
  soOutLineSwitch=new SoSwitch;
  soOutLineSwitch->ref(); // add to scene must be done by derived class
  if(dynamic_cast<CompoundRigidBody*>(parentItem)==0)
    soOutLineSwitch->whichChild.setValue(SO_SWITCH_ALL);
  else
    soOutLineSwitch->whichChild.connectFrom(&((Body*)parentItem)->soOutLineSwitch->whichChild);
  soOutLineSep=new SoSeparator;
  soOutLineSwitch->addChild(soOutLineSep);
  SoLightModel *lm=new SoLightModel;
  soOutLineSep->addChild(lm);
  lm->model.setValue(SoLightModel::BASE_COLOR);
  SoBaseColor *color=new SoBaseColor;
  soOutLineSep->addChild(color);
  color->rgb.setValue(0,0,0);
  SoDrawStyle *style=new SoDrawStyle;
  style->style.setValue(SoDrawStyle::LINES);
  soOutLineSep->addChild(style);

  if(dynamic_cast<CompoundRigidBody*>(parentItem)==0) {
    // draw method
    drawStyle=new SoDrawStyle;
    soSep->addChild(drawStyle);
  
    // GUI
    // draw outline action
    outLine=new QAction(QIconCached(":/outline.svg"),"Draw Out-Line", this);
    outLine->setCheckable(true);
    outLine->setChecked(true);
    outLine->setObjectName("Body::outLine");
    connect(outLine,SIGNAL(changed()),this,SLOT(outLineSlot()));
    // draw method action
    drawMethod=new QActionGroup(this);
    drawMethodPolygon=new QAction(QIconCached(":/filled.svg"),"Draw Style: Filled", drawMethod);
    drawMethodLine=new QAction(QIconCached(":/lines.svg"),"Draw Style: Lines", drawMethod);
    drawMethodPoint=new QAction(QIconCached(":/points.svg"),"Draw Style: Points", drawMethod);
    drawMethodPolygon->setCheckable(true);
    drawMethodPolygon->setData(QVariant(filled));
    drawMethodPolygon->setObjectName("Body::drawMethodPolygon");
    drawMethodLine->setCheckable(true);
    drawMethodLine->setData(QVariant(lines));
    drawMethodLine->setObjectName("Body::drawMethodLine");
    drawMethodPoint->setCheckable(true);
    drawMethodPoint->setData(QVariant(points));
    drawMethodPoint->setObjectName("Body::drawMethodPoint");
    drawMethodPolygon->setChecked(true);
    connect(drawMethod,SIGNAL(triggered(QAction*)),this,SLOT(drawMethodSlot(QAction*)));
  }
}

void Body::frameSensorCB(void *data, SoSensor*) {
  Body* me=(Body*)data;
  double time=0;
  if(me->drawThisPath) 
    time=me->update();
  if(timeUpdater==me)
    MainWindow::getInstance()->setTime(time);
}

QMenu* Body::createMenu() {
  QMenu* menu=Object::createMenu();
  menu->addSeparator()->setText("Properties from: Body");
  menu->addAction(outLine);
  menu->addSeparator();
  menu->addAction(drawMethodPolygon);
  menu->addAction(drawMethodLine);
  menu->addAction(drawMethodPoint);
  return menu;
}

void Body::outLineSlot() {
  if(outLine->isChecked())
    soOutLineSwitch->whichChild.setValue(SO_SWITCH_ALL);
  else
    soOutLineSwitch->whichChild.setValue(SO_SWITCH_NONE);
}

void Body::drawMethodSlot(QAction* action) {
  DrawStyle ds=(DrawStyle)action->data().toInt();
  if(ds==filled)
    drawStyle->style.setValue(SoDrawStyle::FILLED);
  else if(ds==lines)
    drawStyle->style.setValue(SoDrawStyle::LINES);
  else
    drawStyle->style.setValue(SoDrawStyle::POINTS);
}

// number of rows / dt
void Body::resetAnimRange(int numOfRows, double dt) {
  if(numOfRows-1<MainWindow::getInstance()->getTimeSlider()->maximum() || !existFiles) {
    MainWindow::getInstance()->getTimeSlider()->setMaximum(numOfRows-1);
    if(existFiles) {
      QString str("WARNING! Resetting maximal frame number!");
      MainWindow::getInstance()->statusBar()->showMessage(str, 10000);
      cout<<str.toStdString()<<endl;
    }
  }
  if(MainWindow::getInstance()->getDeltaTime()!=dt || !existFiles) {
    MainWindow::getInstance()->getDeltaTime()=dt;
    if(existFiles) {
      QString str("WARNING! dt in HDF5 datas are not the same!");
      MainWindow::getInstance()->statusBar()->showMessage(str, 10000);
      cout<<str.toStdString()<<endl;
    }
  }
  timeUpdater=this;
  existFiles=true;
}

// convenience: convert e.g. "[3;7;7.9]" to std::vector<double>(3,7,7.9)
vector<double> Body::toVector(string str) {
  for(size_t i=0; i<str.length(); i++)
    if(str[i]=='[' || str[i]==']' || str[i]==';') str[i]=' ';
  stringstream stream(str);
  double d;
  vector<double> ret;
  while(1) {
    stream>>d;
    if(stream.fail()) break;
    ret.push_back(d);
  }
  return ret;
}

// convenience: convert e.g. "[3,7;9,7.9]" to std::vector<std::vector<double> >
vector<vector<double> > Body::toMatrix(string str) {
  vector<vector<double> > ret;
  for(size_t i=0; i<str.length(); i++)
    if(str[i]=='[' || str[i]==']' || str[i]==',') str[i]=' ';
  bool br=false;
  while(1) {
    int end=str.find(';'); if(end<0) { end=str.length(); br=true; }
    ret.push_back(toVector(str.substr(0,end)));
    if(br) break;
    str=str.substr(end+1);
  }
  return ret;
}

// convenience: create frame so
SoSeparator* Body::soFrame(double size, double offset, bool pickBBoxAble, SoScale *&scale) {
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


// tess
void Body::tessBeginCB(GLenum type, void *data) {
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

void Body::tessVertexCB(GLdouble *vertex) {
  tessCoord->point.set1Value(tessNumVertices++, vertex[0], vertex[1], vertex[2]);
}

void Body::tessEndCB(void) {
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

SbRotation Body::cardan2Rotation(const SbVec3f &c) {
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

SbVec3f Body::rotation2Cardan(const SbRotation& R) {
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




void Body::triangleCB(void *data, SoCallbackAction *action, const SoPrimitiveVertex *vp1, const SoPrimitiveVertex *vp2, const SoPrimitiveVertex *vp3) {
  Edges* edges=(Edges*)data;
  SbVec3f v1=vp1->getPoint();
  SbVec3f v2=vp2->getPoint();
  SbVec3f v3=vp3->getPoint();
  // convert to coordinates of the action
  SbMatrix mm=action->getModelMatrix();
  mm.multVecMatrix(v1, v1);
  mm.multVecMatrix(v2, v2);
  mm.multVecMatrix(v3, v3);
  ///////////// Round
//  float x,y,z;
//  int K=10000;
//  v1.getValue(x,y,z);
//  x=round(x*K)/K;
//  y=round(y*K)/K;
//  z=round(z*K)/K;
//  v1.setValue(x,y,z);
//  v2.getValue(x,y,z);
//  x=round(x*K)/K;
//  y=round(y*K)/K;
//  z=round(z*K)/K;
//  v2.setValue(x,y,z);
//  v3.getValue(x,y,z);
//  x=round(x*K)/K;
//  y=round(y*K)/K;
//  z=round(z*K)/K;
//  v3.setValue(x,y,z);
  /////////////
  edges->faceVertex.append(edges->vertex.addPoint(v1));
  edges->faceVertex.append(edges->vertex.addPoint(v2));
  edges->faceVertex.append(edges->vertex.addPoint(v3));
  SbVec3f n=(v2-v1).cross(v3-v1);
  n.normalize();
  edges->normal.append(new SbVec3f(n));
}

SoCoordinate3* Body::preCalculateEdges(SoGroup *sep, Edges *edges) {
  SoCallbackAction cba;
  cba.addTriangleCallback(SoShape::getClassTypeId(), triangleCB, edges);
  cba.apply(sep);
  int nr=0;
  for(int i=0; i<edges->faceVertex.getLength()/3; i++) {
    bool foundNeighbour01=false, foundNeighbour12=false, foundNeighbour20=false;
    for(int j=0; j<edges->faceVertex.getLength()/3; j++) {
      if(i==j) continue;
      if((edges->faceVertex[3*i+0]==edges->faceVertex[3*j+0] || edges->faceVertex[3*i+0]==edges->faceVertex[3*j+1] || edges->faceVertex[3*i+0]==edges->faceVertex[3*j+2]) &&
         (edges->faceVertex[3*i+1]==edges->faceVertex[3*j+0] || edges->faceVertex[3*i+1]==edges->faceVertex[3*j+1] || edges->faceVertex[3*i+1]==edges->faceVertex[3*j+2])) {
        edges->innerEdge.append(edges->faceVertex[3*i+0]);
        edges->innerEdge.append(edges->faceVertex[3*i+1]);
        edges->innerEdge.append(i);
        edges->innerEdge.append(j);
        foundNeighbour01=true;
      }
      if((edges->faceVertex[3*i+1]==edges->faceVertex[3*j+0] || edges->faceVertex[3*i+1]==edges->faceVertex[3*j+1] || edges->faceVertex[3*i+1]==edges->faceVertex[3*j+2]) &&
         (edges->faceVertex[3*i+2]==edges->faceVertex[3*j+0] || edges->faceVertex[3*i+2]==edges->faceVertex[3*j+1] || edges->faceVertex[3*i+2]==edges->faceVertex[3*j+2])) {
        edges->innerEdge.append(edges->faceVertex[3*i+1]);
        edges->innerEdge.append(edges->faceVertex[3*i+2]);
        edges->innerEdge.append(i);
        edges->innerEdge.append(j);
        foundNeighbour12=true;
      }
      if((edges->faceVertex[3*i+2]==edges->faceVertex[3*j+0] || edges->faceVertex[3*i+2]==edges->faceVertex[3*j+1] || edges->faceVertex[3*i+2]==edges->faceVertex[3*j+2]) &&
         (edges->faceVertex[3*i+0]==edges->faceVertex[3*j+0] || edges->faceVertex[3*i+0]==edges->faceVertex[3*j+1] || edges->faceVertex[3*i+0]==edges->faceVertex[3*j+2])) {
        edges->innerEdge.append(edges->faceVertex[3*i+2]);
        edges->innerEdge.append(edges->faceVertex[3*i+0]);
        edges->innerEdge.append(i);
        edges->innerEdge.append(j);
        foundNeighbour20=true;
      }
    }
    if(foundNeighbour01==false) {
      edges->boundaryEdge.set1Value(nr++, edges->faceVertex[3*i+0]);
      edges->boundaryEdge.set1Value(nr++, edges->faceVertex[3*i+1]);
      edges->boundaryEdge.set1Value(nr++, -1);
    }
    if(foundNeighbour12==false) {
      edges->boundaryEdge.set1Value(nr++, edges->faceVertex[3*i+1]);
      edges->boundaryEdge.set1Value(nr++, edges->faceVertex[3*i+2]);
      edges->boundaryEdge.set1Value(nr++, -1);
    }
    if(foundNeighbour20==false) {
      edges->boundaryEdge.set1Value(nr++, edges->faceVertex[3*i+2]);
      edges->boundaryEdge.set1Value(nr++, edges->faceVertex[3*i+0]);
      edges->boundaryEdge.set1Value(nr++, -1);
    }
  }
  SoCoordinate3 *soEdgeVertex=new SoCoordinate3;
  soEdgeVertex->point.setValuesPointer(edges->vertex.numPoints(), edges->vertex.getPointsArrayPtr());
  return soEdgeVertex;
}

SoIndexedLineSet* Body::calculateCreaseEdges(double creaseAngle, Edges *edges) {
  SoIndexedLineSet *soCreaseEdge=new SoIndexedLineSet;
  float CREASEANGLE=creaseAngle;
  int nr=0;
  for(int i=0; i<edges->innerEdge.getLength()/4; i++) {
    int v1=edges->innerEdge[4*i+0];
    int v2=edges->innerEdge[4*i+1];
    int f1=edges->innerEdge[4*i+2];
    int f2=edges->innerEdge[4*i+3];
    if(edges->normal[f1]->dot(*edges->normal[f2])<cos(CREASEANGLE)) {
      soCreaseEdge->coordIndex.set1Value(nr++, v1);
      soCreaseEdge->coordIndex.set1Value(nr++, v2);
      soCreaseEdge->coordIndex.set1Value(nr++, -1);
    }
  }
  return soCreaseEdge;
}

SoIndexedLineSet* Body::calculateBoundaryEdges(Edges *edges) {
  SoIndexedLineSet *soBoundaryEdge=new SoIndexedLineSet;
  soBoundaryEdge->coordIndex.copyFrom(edges->boundaryEdge);
  return soBoundaryEdge;
}
