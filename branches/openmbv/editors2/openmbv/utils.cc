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
#include <Inventor/nodes/SoLineSet.h>
#ifdef HAVE_UNORDERED_MAP
#  include <unordered_map>
#else
#  include <map>
#  define unordered_map map
#endif
#include "SoSpecial.h"
#include <iostream>
#include <QtGui/QDialog>
#include <QtGui/QPushButton>
#include <QtGui/QLineEdit>
#include <QtGui/QMessageBox>
#include <QtGui/QGridLayout>
#include <QtGui/QComboBox>
#include <QtGui/QLabel>

using namespace std;

bool Utils::initialized=false;

void Utils::initialize() {
  if(initialized==true) return;
  initialized=true;

  // tess
  gluTessCallback(tess, GLU_TESS_BEGIN_DATA, (void (CALLMETHOD *)())tessBeginCB);
  gluTessCallback(tess, GLU_TESS_VERTEX, (void (CALLMETHOD *)())tessVertexCB);
  gluTessCallback(tess, GLU_TESS_END, (void (CALLMETHOD *)())tessEndCB);
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
    if(in.openFile(filename.c_str(), true)) { // if file can be opened, read it
      ins.first->second=SoDB::readAll(&in); // stored in a global cache => false positive in valgrind
      ins.first->second->ref(); // increment reference count to prevent a delete for cached entries
      return ins.first->second;
    }
    else { // open failed, remove from cache and return a empty IV
      cout<<"ERROR: Unable to find IV file "<<filename<<"."<<endl;
      myIvBodyCache.erase(ins.first);
      return new SoSeparator;
    }
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

OpenMBV::Object *Utils::createObjectEditor(const vector<FactoryElement> &factory, const vector<string> &existingNames, const string &title) {
  bool exist;
  int i=0;
  string name;
  do {
    i++;
    stringstream str;
    str<<"Untitled"<<i;
    name=str.str();
    exist=false;
    for(unsigned int j=0; j<existingNames.size(); j++)
      if(existingNames[j]==name) {
        exist=true;
        break;
      }
  } while(exist==true);

  QDialog dialog;
  dialog.setWindowTitle(title.c_str());
  QGridLayout *layout=new QGridLayout();
  dialog.setLayout(layout);

  layout->addWidget(new QLabel("Type:"), 0, 0);
  QComboBox *cb=new QComboBox();
  layout->addWidget(cb, 0, 1);
  for(unsigned int i=0; i<factory.size(); i++)
    cb->addItem(factory[i].get<0>(), factory[i].get<1>().c_str());

  layout->addWidget(new QLabel("Name:"), 1, 0);
  QLineEdit *lineEdit=new QLineEdit();
  layout->addWidget(lineEdit, 1, 1);
  lineEdit->setText(name.c_str());

  QPushButton *cancel=new QPushButton("Cancel");
  layout->addWidget(cancel, 2, 0);
  QObject::connect(cancel, SIGNAL(released()), &dialog, SLOT(reject()));
  QPushButton *ok=new QPushButton("OK");
  layout->addWidget(ok, 2, 1);
  ok->setDefault(true);
  QObject::connect(ok, SIGNAL(released()), &dialog, SLOT(accept()));

  bool unique;
  do {
    if(dialog.exec()!=QDialog::Accepted) return NULL;
    unique=true;
    for(unsigned int j=0; j<existingNames.size(); j++)
      if(existingNames[j]==lineEdit->text().toStdString()) {
        QMessageBox::information(0, "Information", "The entered name already exists!");
        unique=false;
        break;
      }
  } while(!unique);

  OpenMBV::Object *obj=factory[cb->currentIndex()].get<2>()();
  obj->setName(lineEdit->text().toStdString());
  return obj;
}
