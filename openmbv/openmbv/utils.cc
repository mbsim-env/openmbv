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

#include <config.h>
#include "utils.h"
#include <Inventor/nodes/SoLineSet.h>
#include "SoSpecial.h"
#include "mainwindow.h"
#include <iostream>
#include <QDialog>
#include <QPushButton>
#include <QLineEdit>
#include <QMessageBox>
#include <QGridLayout>
#include <QComboBox>
#include <QLabel>
#include <boost/dll.hpp>

using namespace std;

namespace OpenMBVGUI {

unordered_map<string, Utils::SoDeleteSeparator> Utils::ivBodyCache;
unordered_map<string, QIcon> Utils::iconCache;
bool Utils::initialized=false;

void Utils::initialize() {
  if(initialized) return;
  initialized=true;

  // tess
  gluTessCallback(tess, GLU_TESS_BEGIN_DATA, (void (CALLMETHOD *)())tessBeginCB);
  gluTessCallback(tess, GLU_TESS_VERTEX, (void (CALLMETHOD *)())tessVertexCB);
  gluTessCallback(tess, GLU_TESS_END, (void (CALLMETHOD *)())tessEndCB);
}

void Utils::deinitialize() {
  if(!initialized) return;
  initialized=false;

  iconCache.clear();
}

const QIcon& Utils::QIconCached(string filename) {
  // fix relative filename
  if(filename[0]!=':' && filename[0]!='/')
    filename=getIconPath()+"/"+filename;
  
  pair<unordered_map<string, QIcon>::iterator, bool> ins=iconCache.insert(pair<string, QIcon>(filename, QIcon()));
  if(ins.second)
    return ins.first->second=QIcon(filename.c_str());
  return ins.first->second;
}

SoSeparator* Utils::SoDBreadAllCached(const string &filename) {
  auto ins=ivBodyCache.emplace(filename, SoDeleteSeparator());
  if(ins.second) {
    SoInput in;
    if(in.openFile(filename.c_str(), true)) { // if file can be opened, read it
      ins.first->second.s=SoDB::readAll(&in); // stored in a global cache => false positive in valgrind
      ins.first->second.s->ref(); // increment reference count to prevent a delete for cached entries
      return ins.first->second.s;
    }
    else { // open failed, remove from cache and return a empty IV
      QString str("Unable to find IV file %1."); str=str.arg(filename.c_str());
      MainWindow::getInstance()->statusBar()->showMessage(str);
      msgStatic(Warn)<<str.toStdString()<<endl;
      ivBodyCache.erase(ins.first);
      return new SoSeparator;
    }
  }
  return ins.first->second.s;
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
  auto *coord=new SoCoordinate3;
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
  return {a,b,g};
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
  auto *parent=(SoGroup*)data;
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

void Utils::tessEndCB() {
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

std::shared_ptr<OpenMBV::Object> Utils::createObjectEditor(const vector<FactoryElement> &factory, const vector<string> &existingNames, const string &title) {
  bool exist;
  int i=0;
  string name;
  do {
    i++;
    stringstream str;
    str<<"Untitled"<<i;
    name=str.str();
    exist=false;
    for(const auto & existingName : existingNames)
      if(existingName==name) {
        exist=true;
        break;
      }
  } while(exist);

  QDialog dialog;
  dialog.setWindowTitle(title.c_str());
  auto *layout=new QGridLayout();
  dialog.setLayout(layout);

  layout->addWidget(new QLabel("Type:"), 0, 0);
  auto *cb=new QComboBox();
  layout->addWidget(cb, 0, 1);
  for(const auto & i : factory)
    cb->addItem(get<0>(i), get<1>(i).c_str());

  layout->addWidget(new QLabel("Name:"), 1, 0);
  auto *lineEdit=new QLineEdit();
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
    if(dialog.exec()!=QDialog::Accepted) return std::shared_ptr<OpenMBV::Object>();
    unique=true;
    for(const auto & existingName : existingNames)
      if(existingName==lineEdit->text().toStdString()) {
        QMessageBox::information(nullptr, "Information", "The entered name already exists!");
        unique=false;
        break;
      }
  } while(!unique);

  std::shared_ptr<OpenMBV::Object> obj=get<2>(factory[cb->currentIndex()])();
  obj->setName(lineEdit->text().toStdString());
  return obj;
}

namespace {
  boost::filesystem::path sharePath(boost::dll::program_location().parent_path().parent_path()/"share");
}

string Utils::getIconPath() {
  return (sharePath/"openmbv"/"icons").string();
}

string Utils::getXMLDocPath() {
  return (sharePath/"mbxmlutils"/"doc").string();
}

string Utils::getDocPath() {
  return (sharePath/"openmbv"/"doc").string();
}

}
