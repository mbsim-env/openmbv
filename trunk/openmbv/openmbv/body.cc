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

using namespace std;

bool Body::existFiles=false;
Body *Body::timeUpdater=0;
double Body::eps;

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
    sensor->setPriority(0);
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
    outLine=new QAction(QIcon(":/outline.svg"),"Draw Out-Line", this);
    outLine->setCheckable(true);
    outLine->setChecked(true);
    outLine->setObjectName("Body::outLine");
    connect(outLine,SIGNAL(changed()),this,SLOT(outLineSlot()));
    // draw method action
    drawMethod=new QActionGroup(this);
    drawMethodPolygon=new QAction(QIcon(":/filled.svg"),"Draw Style: Filled", drawMethod);
    drawMethodLine=new QAction(QIcon(":/lines.svg"),"Draw Style: Lines", drawMethod);
    drawMethodPoint=new QAction(QIcon(":/points.svg"),"Draw Style: Points", drawMethod);
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
  double time;
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
  for(int i=0; i<str.length(); i++)
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
  for(int i=0; i<str.length(); i++)
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



// combine, calcaulate vertex/normal

bool Body::SbVec3fHash::operator()(const SbVec3f& v1, const SbVec3f& v2) const {
  float x1,y1,z1,x2,y2,z2;
  v1.getValue(x1,y1,z1);
  v2.getValue(x2,y2,z2);
  if(x1<x2-eps) return true;
  else if(x1>x2+eps) return false;
  else
    if(y1<y2-eps) return true;
    else if(y1>y2+eps) return false;
    else
      if(z1<z2-eps) return true;
      else if(z1>z2+eps) return false;
      else return false;
};

void Body::combine(const SoMFVec3f& v, SoMFVec3f& newv, SoMFInt32& newvi) {
  map<SbVec3f, int, SbVec3fHash> hash;
  for(int i=0; i<v.getNum(); i++) {
    int &r=hash[*(v.getValues(i))]; // add vertex to a hash map
    // if vertex not exist in hash map copy it to newvv,
    // set corospondenting newvi,
    // and set hash map value to the index of this vertex
    if(r==0) {
      newv.set1Value(newv.getNum(), *v.getValues(i));
      newvi.set1Value(i, newv.getNum()-1);
      r=newv.getNum()-1+1;
    }
    // if vertix exist in hash map,
    // set corrospondenting newvi to the value in the hash map
    else
      newvi.set1Value(i, r-1);
  }
}

void Body::convertIndex(SoMFInt32& fvi, const SoMFInt32& newvi) {
  for(int i=0; i<fvi.getNum(); i++)
    if(fvi[i]>=0) fvi.set1Value(i, newvi[fvi[i]]);
}

// complexibility: ?
bool Body::TwoIndexHash::operator()(const TwoIndex& l1, const TwoIndex& l2) const {
  if(l1.vi1<l2.vi1) return true;
  else if(l1.vi1>l2.vi1) return false;
  else
    if(l1.vi2<l2.vi2) return true;
    else if(l1.vi2>l2.vi2) return false;
    else return false;
}

void Body::computeNormals(const SoMFInt32& fvi, const SoMFVec3f &v, SoMFInt32& fni, SoMFVec3f& n, SoMFInt32& oli, double smoothBarrier) {
  map<TwoIndex, vector<XXX>, TwoIndexHash> lni; // line normal index = index of normal of start/end point
  map<int, vector<int> > vni;

  for(int i=0; i<fvi.getNum(); i+=4) {
    // set face normals fn
    SbVec3f fn, ft, fb;
    ft=v[fvi[i+1]]-v[fvi[i+0]];
    fb=v[fvi[i+2]]-v[fvi[i+0]];
    fn=ft.cross(fb);
    fni.set1Value(i+0, i+0);
    fni.set1Value(i+1, i+1);
    fni.set1Value(i+2, i+2);
    fni.set1Value(i+3, -1);
    n.set1Value(i+0, fn);
    n.set1Value(i+1, fn);
    n.set1Value(i+2, fn);

    // store all face indexies of each line in a map
    TwoIndex l;
    XXX xxx;
    l.vi1=fvi[i+0]; l.vi2=fvi[i+1];
    xxx.ni1=i+0; xxx.ni2=i+1;
    if(l.vi1>l.vi2) { int dummy=l.vi1; l.vi1=l.vi2; l.vi2=dummy;
                          dummy=xxx.ni1; xxx.ni1=xxx.ni2; xxx.ni2=dummy; }
    lni[l].push_back(xxx);
    l.vi1=fvi[i+1]; l.vi2=fvi[i+2];
    xxx.ni1=i+1; xxx.ni2=i+2;
    if(l.vi1>l.vi2) { int dummy=l.vi1; l.vi1=l.vi2; l.vi2=dummy;
                          dummy=xxx.ni1; xxx.ni1=xxx.ni2; xxx.ni2=dummy; }
    lni[l].push_back(xxx);
    l.vi1=fvi[i+2]; l.vi2=fvi[i+0];
    xxx.ni1=i+2; xxx.ni2=i+0;
    if(l.vi1>l.vi2) { int dummy=l.vi1; l.vi1=l.vi2; l.vi2=dummy;
                          dummy=xxx.ni1; xxx.ni1=xxx.ni2; xxx.ni2=dummy; }
    lni[l].push_back(xxx);

    // vni
    vni[fvi[i+0]].push_back(i+0);
    vni[fvi[i+1]].push_back(i+1);
    vni[fvi[i+2]].push_back(i+2);
  }
  map<TwoIndex, vector<XXX>, TwoIndexHash>::iterator i;
  int ni1, ni2;
  SbVec3f nNew;
  for(i=lni.begin(); i!=lni.end(); i++) {
    if(i->second.size()!=2) continue;
    bool smooth=false;
    ni1=i->second[0].ni1; ni2=i->second[1].ni1;
    if(acos(n[fni[ni1]].dot(n[fni[ni2]])/n[fni[ni1]].length()/n[fni[ni2]].length())<smoothBarrier) {
      smooth=true;
      nNew=n[fni[ni1]]+n[fni[ni2]];
      n.set1Value(fni[ni1], nNew); n.set1Value(fni[ni2], nNew);
      vector<int> vvv=vni[i->first.vi1];
      for(int k=0; k<vvv.size(); k++)
        if(fni[vvv[k]]==fni[ni2]) fni.set1Value(vvv[k], fni[ni1]);
    }
    ni1=i->second[0].ni2; ni2=i->second[1].ni2;
    if(acos(n[fni[ni1]].dot(n[fni[ni2]])/n[fni[ni1]].length()/n[fni[ni2]].length())<smoothBarrier) {
      smooth=true;
      nNew=n[fni[ni1]]+n[fni[ni2]];
      n.set1Value(fni[ni1], nNew); n.set1Value(fni[ni2] ,nNew);
      vector<int> vvv=vni[i->first.vi1];
      for(int k=0; k<vvv.size(); k++)
        if(fni[vvv[k]]==fni[ni2]) fni.set1Value(vvv[k], fni[ni1]);
    }
    if(!smooth) {
      oli.set1Value(oli.getNum(), fvi[i->second[0].ni1]);
      oli.set1Value(oli.getNum(), fvi[i->second[0].ni2]);
      oli.set1Value(oli.getNum(), -1);
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
