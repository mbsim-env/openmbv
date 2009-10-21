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
#include "arrow.h"
#include "mainwindow.h"
#include <Inventor/nodes/SoCone.h>
#include <Inventor/nodes/SoCylinder.h>
#include <Inventor/nodes/SoBaseColor.h>

using namespace std;

Arrow::Arrow(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : DynamicColoredBody(element, h5Parent, parentItem, soParent) {
  iconFile=":/arrow.svg";
  setIcon(0, QIcon(iconFile.c_str()));

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
  TiXmlElement *e=element->FirstChildElement(OPENMBVNS"diameter");
  double diameter=toVector(e->GetText())[0];
  e=e->NextSiblingElement();
  double headDiameter=toVector(e->GetText())[0];
  e=e->NextSiblingElement();
  headLength=toVector(e->GetText())[0];
  e=e->NextSiblingElement();
  string type_=string(e->GetText()).substr(1,string(e->GetText()).length()-2);
  if(type_=="line") type=line;
  else if(type_=="fromHead") type=fromHead;
  else if(type_=="toHead") type=toHead;
  else type=bothHeads;
  if(type!=toHead) printf("Only toHead is implemented yet!!!\n"); //TODO
  e=e->NextSiblingElement();
  scaleLength=toVector(e->GetText())[0];

  // create so
  // path
  soPathSwitch=new SoSwitch;
  soSep->addChild(soPathSwitch);
  SoSeparator *pathSep=new SoSeparator;
  soPathSwitch->addChild(pathSep);
  SoBaseColor *col=new SoBaseColor;
  col->rgb.setValue(0, 1, 0);
  pathSep->addChild(col);
  pathCoord=new SoCoordinate3;
  pathSep->addChild(pathCoord);
  pathLine=new SoLineSet;
  pathSep->addChild(pathLine);
  soPathSwitch->whichChild.setValue(SO_SWITCH_NONE);
  pathMaxFrameRead=-1;

  // Arrow
  // mat
  mat=new SoMaterial;
  soSep->addChild(mat);
  if(!isnan(staticColor)) setColor(mat, staticColor);
  // translate to To-Point
  toPoint=new SoTranslation;
  soSep->addChild(toPoint);
  // rotation to dx,dy,dz
  rotation1=new SoRotation;
  soSep->addChild(rotation1);
  rotation2=new SoRotation;
  soSep->addChild(rotation2);
  // full scale (l<2*headLength)
  scale1=new SoScale;
  soSep->addChild(scale1);
  // outline
  soSep->addChild(soOutLineSwitch);
  // trans1
  SoTranslation *trans1=new SoTranslation;
  soSep->addChild(trans1);
  trans1->translation.setValue(0, -headLength/2, 0);
  // cone
  SoCone *cone1=new SoCone;
  soSep->addChild(cone1);
  cone1->bottomRadius.setValue(headDiameter/2);
  cone1->height.setValue(headLength);
  // trans2
  SoTranslation *trans2=new SoTranslation;
  soSep->addChild(trans2);
  trans2->translation.setValue(0, -headLength/2, 0);
  // scale only arrow not head (l>2*headLength)
  scale2=new SoScale;
  soSep->addChild(scale2);
  // trans2
  SoTranslation *trans3=new SoTranslation;
  soSep->addChild(trans3);
  trans3->translation.setValue(0, -headLength/2, 0);
  // cylinder
  SoCylinder *cylinder=new SoCylinder;
  soSep->addChild(cylinder);
  cylinder->radius.setValue(diameter/2);
  cylinder->height.setValue(headLength);

  // outline
  SoTranslation *transLO1=new SoTranslation;
  soOutLineSep->addChild(transLO1);
  transLO1->translation.setValue(0, -headLength, 0);
  SoCylinder *coneOL1=new SoCylinder;
  soOutLineSep->addChild(coneOL1);
  coneOL1->height.setValue(0);
  coneOL1->radius.setValue(headDiameter/2);
  coneOL1->parts.setValue(SoCylinder::SIDES);
  SoCylinder *coneOL2=new SoCylinder;
  soOutLineSep->addChild(coneOL2);
  coneOL2->height.setValue(0);
  coneOL2->radius.setValue(diameter/2);
  coneOL2->parts.setValue(SoCylinder::SIDES);
  soOutLineSep->addChild(scale2);
  SoTranslation *transLO2=new SoTranslation;
  soOutLineSep->addChild(transLO2);
  transLO2->translation.setValue(0, -headLength, 0);
  soOutLineSep->addChild(coneOL2);
 
  // GUI
  path=new QAction(QIcon(":/path.svg"),"Draw Path of To-Point", this);
  path->setCheckable(true);
  path->setObjectName("Arrow::path");
  connect(path,SIGNAL(changed()),this,SLOT(pathSlot()));
}

double Arrow::update() {
  if(h5Group==0) return 0;

  int frame=MainWindow::getInstance()->getFrame()->getValue();
  // read from hdf5
  data=h5Data->getRow(frame);
  
  // set scene values
  double dx=data[4], dy=data[5], dz=data[6];
  length=sqrt(dx*dx+dy*dy+dz*dz)*scaleLength;

  // path
  if(path->isChecked()) {
    for(int i=pathMaxFrameRead+1; i<=frame; i++) {
      vector<double> localData=h5Data->getRow(i);
      if(length<1e-10)
        if(i-1<0)
          pathCoord->point.set1Value(i, 0, 0, 0); // dont known what coord to write; using 0
        else
          pathCoord->point.set1Value(i, *pathCoord->point.getValues(i-1));
      else
        pathCoord->point.set1Value(i, localData[1], localData[2], localData[3]);
    }
    pathMaxFrameRead=frame;
    pathLine->numVertices.setValue(1+frame);
  }
  
  if(draw->isChecked())
    if(length<1e-10) {
      soSwitch->whichChild.setValue(SO_SWITCH_NONE);
      soBBoxSwitch->whichChild.setValue(SO_SWITCH_NONE);
      return data[0];
    }
    else {
      soSwitch->whichChild.setValue(SO_SWITCH_ALL);
      if(bbox->isChecked()) soBBoxSwitch->whichChild.setValue(SO_SWITCH_ALL);
    }

  // scale factors
  if(length<2*headLength)
    scale1->scaleFactor.setValue(length/2/headLength,length/2/headLength,length/2/headLength);
  else
    scale1->scaleFactor.setValue(1,1,1);
  if(length>2*headLength)
    scale2->scaleFactor.setValue(1,(length-headLength)/headLength,1);
  else
    scale2->scaleFactor.setValue(1,1,1);
  // trans to To-Point
  toPoint->translation.setValue(data[1], data[2], data[3]);
  // rotation to dx,dy,dz
  rotation1->rotation.setValue(SbVec3f(0, 0, 1), -atan2(dx,dy));
  rotation2->rotation.setValue(SbVec3f(1, 0, 0), atan2(dz,sqrt(dx*dx+dy*dy)));
  // mat
  if(isnan(staticColor)) setColor(mat, data[7]);

  return data[0];
}

QString Arrow::getInfo() {
  if(data.size()==0) update();
  return DynamicColoredBody::getInfo()+
         QString("-----<br/>")+
         QString("<b>To-Point:</b> %1, %2, %3<br/>").arg(data[1]).arg(data[2]).arg(data[3])+
         QString("<b>Vector:</b> %1, %2, %3<br/>").arg(data[4]).arg(data[5]).arg(data[6])+
         QString("<b>Length:</b> %1<br/>").arg(length/scaleLength);
}

void Arrow::pathSlot() {
  if(path->isChecked()) {
    soPathSwitch->whichChild.setValue(SO_SWITCH_ALL);
    update();
  }
  else
    soPathSwitch->whichChild.setValue(SO_SWITCH_NONE);
}

QMenu* Arrow::createMenu() {
  QMenu* menu=DynamicColoredBody::createMenu();
  menu->addSeparator()->setText("Properties from: Arrow");
  menu->addAction(path);
  return menu;
}
