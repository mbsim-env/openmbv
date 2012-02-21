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
#include "utils.h"
#include "openmbvcppinterface/arrow.h"
#include <Inventor/engines/SoCalculator.h>
#include <cfloat>

using namespace std;

Arrow::Arrow(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : DynamicColoredBody(obj, parentItem, soParent, ind) {
  arrow=(OpenMBV::Arrow*)obj;
  iconFile=":/arrow.svg";
  setIcon(0, Utils::QIconCached(iconFile.c_str()));

  //h5 dataset
  int rows=arrow->getRows();
  double dt;
  if(rows>=2) dt=arrow->getRow(1)[0]-arrow->getRow(0)[0]; else dt=0;
  resetAnimRange(rows, dt);

  // read XML and calcualte all expressions for the Arrow
  #define diameter calc->a
  #define headLength calc->b
  #define headDiameter calc->c
  #define diameterHalf calc->oa
  #define headDiameterHalf calc->ob
  #define dTranslateVecY calc->oA // [0, +headLength*(headDiameter-diameter)/headDiameter, 0]
  #define negdTranslateVecY calc->oB // -dTranslate
  #define negHeadLengthHalfVecY calc->oC
  #define negHeadLengthVecY calc->oD
  SoCalculator *calc=new SoCalculator;
  diameterEditor=new FloatEditor(this, QIcon(), "Diameter", &calc->a);
  diameterEditor->setRange(0, DBL_MAX);
  diameterEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getDiameter, &OpenMBV::Arrow::setDiameter);
  headLengthEditor=new FloatEditor(this, QIcon(), "Head Length", &calc->b);
  headLengthEditor->setRange(0, DBL_MAX);
  headLengthEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getHeadLength, &OpenMBV::Arrow::setHeadLength);
  connect(headLengthEditor, SIGNAL(valueChanged()), this, SLOT(update()));
  headDiameterEditor=new FloatEditor(this, QIcon(), "Head Diameter", &calc->c);
  headDiameterEditor->setRange(0, DBL_MAX);
  headDiameterEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getHeadDiameter, &OpenMBV::Arrow::setHeadDiameter);
  scaleLengthEditor=new FloatEditor(this, QIcon(), "Scale Length", static_cast<SoSFFloat*>(NULL));
  scaleLengthEditor->setRange(0, DBL_MAX);
  scaleLengthEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getScaleLength, &OpenMBV::Arrow::setScaleLength);
  connect(scaleLengthEditor, SIGNAL(valueChanged()), this, SLOT(update()));
  calc->expression.set1Value(0, "oa=a/2");
  calc->expression.set1Value(1, "ob=c/2");
  calc->expression.set1Value(2, "oA=vec3f(0, +b*(c-a)/c, 0)");
  calc->expression.set1Value(3, "oB=vec3f(0, -b*(c-a)/c, 0)");
  calc->expression.set1Value(4, "oC=vec3f(0, -b/2, 0)");
  calc->expression.set1Value(5, "oD=vec3f(0, -b, 0)");

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
  pathMaxFrameRead=-1;


  if(arrow->getType()==OpenMBV::Arrow::line) {
    // Line
 
    // color
    baseColor=new SoBaseColor;
    soSep->addChild(baseColor);
    if(!isnan(staticColor)) setColor(NULL, staticColor, baseColor);
    // line width
    SoDrawStyle *drawStyle=new SoDrawStyle;
    soSep->addChild(drawStyle);
    drawStyle->lineWidth.setValue(2);
    // line
    lineCoord=new SoCoordinate3;
    soSep->addChild(lineCoord);
    SoLineSet *ls=new SoLineSet;
    soSep->addChild(ls);
    ls->numVertices.set1Value(0, 2);
  }
  else {
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
    // arrow separator
    SoSeparator *arrowSep=new SoSeparator;
    soSep->addChild(arrowSep);
    // add arrow twice for type==bothHeads with half length: mirrored by x-z-plane and moved
    if(arrow->getType()==OpenMBV::Arrow::bothHeads || arrow->getType()==OpenMBV::Arrow::bothDoubleHeads) {
      SoScale *bScale=new SoScale;
      soSep->addChild(bScale);
      bScale->scaleFactor.setValue(1,-1,1);
      bTrans=new SoTranslation;
      soSep->addChild(bTrans);
      soSep->addChild(arrowSep);
    }
    // full scale (l<2*headLength)
    scale1=new SoScale;
    arrowSep->addChild(scale1);
    // outline
    arrowSep->addChild(soOutLineSwitch);
    // trans1
    SoTranslation *trans1=new SoTranslation;
    arrowSep->addChild(trans1);
    trans1->translation.connectFrom(&negHeadLengthHalfVecY);
    // cone
    SoCone *cone1=new SoCone;
    arrowSep->addChild(cone1);
    cone1->bottomRadius.connectFrom(&headDiameterHalf);
    cone1->height.connectFrom(&headLength);
    // add the head twice for double heads
    SoTranslation *dTrans=NULL;
    if(arrow->getType()==OpenMBV::Arrow::fromDoubleHead ||
       arrow->getType()==OpenMBV::Arrow::toDoubleHead ||
       arrow->getType()==OpenMBV::Arrow::bothDoubleHeads) {
      dTrans=new SoTranslation;
      arrowSep->addChild(dTrans);
      dTrans->translation.connectFrom(&negdTranslateVecY);
      arrowSep->addChild(cone1);
      dTrans=new SoTranslation;
      arrowSep->addChild(dTrans);
      dTrans->translation.connectFrom(&dTranslateVecY);
    }
    // trans2
    SoTranslation *trans2=new SoTranslation;
    arrowSep->addChild(trans2);
    trans2->translation.connectFrom(&negHeadLengthHalfVecY);
    // scale only arrow not head (l>2*headLength)
    scale2=new SoScale;
    arrowSep->addChild(scale2);
    // trans2
    SoTranslation *trans3=new SoTranslation;
    arrowSep->addChild(trans3);
    trans3->translation.connectFrom(&negHeadLengthHalfVecY);
    // cylinder
    SoCylinder *cylinder=new SoCylinder;
    arrowSep->addChild(cylinder);
    cylinder->radius.connectFrom(&diameterHalf);
    cylinder->height.connectFrom(&headLength);

    // outline
    SoTranslation *transLO1=new SoTranslation;
    soOutLineSep->addChild(transLO1);
    transLO1->translation.connectFrom(&negHeadLengthVecY);
    SoCylinder *cylOL1=new SoCylinder;
    soOutLineSep->addChild(cylOL1);
    cylOL1->height.setValue(0);
    cylOL1->radius.connectFrom(&headDiameterHalf);
    cylOL1->parts.setValue(SoCylinder::SIDES);
    SoCylinder *cylOL2=new SoCylinder;
    soOutLineSep->addChild(cylOL2);
    cylOL2->height.setValue(0);
    cylOL2->radius.connectFrom(&diameterHalf);
    cylOL2->parts.setValue(SoCylinder::SIDES);
    // add the head outline twice for double heads
    if(arrow->getType()==OpenMBV::Arrow::fromDoubleHead ||
       arrow->getType()==OpenMBV::Arrow::toDoubleHead ||
       arrow->getType()==OpenMBV::Arrow::bothDoubleHeads) {
      dTrans=new SoTranslation;
      soOutLineSep->addChild(dTrans);
      dTrans->translation.connectFrom(&negdTranslateVecY);
      soOutLineSep->addChild(cylOL1);
      soOutLineSep->addChild(cylOL2);
      dTrans=new SoTranslation;
      soOutLineSep->addChild(dTrans);
      dTrans->translation.connectFrom(&dTranslateVecY);
    }
    // for type==bothHeads do not draw the middle outline circle
    if(arrow->getType()!=OpenMBV::Arrow::bothHeads && arrow->getType()!=OpenMBV::Arrow::bothDoubleHeads) {
      soOutLineSep->addChild(scale2);
      SoTranslation *transLO2=new SoTranslation;
      soOutLineSep->addChild(transLO2);
      transLO2->translation.connectFrom(&negHeadLengthVecY);
      soOutLineSep->addChild(cylOL2);
    }
  }
 
  // GUI
  path=new BoolEditor(this, Utils::QIconCached(":/path.svg"),"Draw Path of To-Point", &soPathSwitch->whichChild);
  path->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getPath, &OpenMBV::Arrow::setPath);
  connect(path->getAction(),SIGNAL(changed()),this,SLOT(update())); // a special action is required by path

  #undef diameter
  #undef headLength
  #undef headDiameter
  #undef diameterHalf
  #undef headDiameterHalf
  #undef dTranslateVecY
  #undef negdTranslateVecY
  #undef negHeadLengthHalfVecY
  #undef negHeadLengthVecY
}

double Arrow::update() {
  int frame=MainWindow::getInstance()->getFrame()->getValue();
  // read from hdf5
  data=arrow->getRow(frame);
 
  // convert data from fromHead representation to toHead representation
  if(arrow->getType()==OpenMBV::Arrow::fromHead || arrow->getType()==OpenMBV::Arrow::fromDoubleHead)
  {
    // to-point_new=to-point_old-dv_old; dv_new=-dv_old
    data[1]-=data[4]*arrow->getScaleLength();
    data[2]-=data[5]*arrow->getScaleLength();
    data[3]-=data[6]*arrow->getScaleLength();
    data[4]*=-1.0;
    data[5]*=-1.0;
    data[6]*=-1.0;
  }

  // set scene values
  double dx=data[4], dy=data[5], dz=data[6];
  length=sqrt(dx*dx+dy*dy+dz*dz)*arrow->getScaleLength();

  // path
  if(path->getAction()->isChecked()) {
    for(int i=pathMaxFrameRead+1; i<=frame; i++) {
      vector<double> localData=arrow->getRow(i);
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

  // if length is 0 do not draw
  if(draw->getAction()->isChecked()) {
    if(length<1e-10) {
      soSwitch->whichChild.setValue(SO_SWITCH_NONE);
      soBBoxSwitch->whichChild.setValue(SO_SWITCH_NONE);
      return data[0];
    }
    else {
      soSwitch->whichChild.setValue(SO_SWITCH_ALL);
      if(bbox->getAction()->isChecked()) soBBoxSwitch->whichChild.setValue(SO_SWITCH_ALL);
    }
  }

  // if type==line: draw update line and exit
  if(arrow->getType()==OpenMBV::Arrow::line) {
    if(isnan(staticColor)) setColor(NULL, data[7], baseColor);
    lineCoord->point.set1Value(0, data[1], data[2], data[3]); // to point
    lineCoord->point.set1Value(1, data[1]-dx*arrow->getScaleLength(), data[2]-dy*arrow->getScaleLength(), data[3]-dz*arrow->getScaleLength()); // from point
    return data[0];
  }

  // type!=line: draw arrow

  // for type==bothHeads: set translation of second arrow and draw two arrows with half length
  if(arrow->getType()==OpenMBV::Arrow::bothHeads || arrow->getType()==OpenMBV::Arrow::bothDoubleHeads) {
    bTrans->translation.setValue(0, length, 0);
    length/=2.0;
  }

  // scale factors
  if(length<2*arrow->getHeadLength())
    scale1->scaleFactor.setValue(length/2/arrow->getHeadLength(),length/2/arrow->getHeadLength(),length/2/arrow->getHeadLength());
  else
    scale1->scaleFactor.setValue(1,1,1);
  if(length>2*arrow->getHeadLength())
    scale2->scaleFactor.setValue(1,(length-arrow->getHeadLength())/arrow->getHeadLength(),1);
  else
    scale2->scaleFactor.setValue(1,1,1);
  // trans to To-Point
  toPoint->translation.setValue(data[1], data[2], data[3]);
  // rotation to dx,dy,dz
  rotation1->rotation.setValue(SbVec3f(0, 0, 1), -atan2(dx,dy));
  rotation2->rotation.setValue(SbVec3f(1, 0, 0), atan2(dz,sqrt(dx*dx+dy*dy)));
  // mat
  if(isnan(staticColor)) setColor(mat, data[7]);

  // for type==bothHeads: reset half length
  if(arrow->getType()==OpenMBV::Arrow::bothHeads || arrow->getType()==OpenMBV::Arrow::bothDoubleHeads)
    length*=2.0;

  return data[0];
}

QString Arrow::getInfo() {
  if(data.size()==0) update();

  // convert data back from toHead to fromHead if type==fromHead or fromDoubleHead
  double drFactor=1;
  double toMove[]={0, 0, 0};
  if(arrow->getType()==OpenMBV::Arrow::fromHead || arrow->getType()==OpenMBV::Arrow::fromDoubleHead)
  {
    drFactor=-1;
    toMove[0]=-data[4]*arrow->getScaleLength();
    toMove[1]=-data[5]*arrow->getScaleLength();
    toMove[2]=-data[6]*arrow->getScaleLength();
  }

  return DynamicColoredBody::getInfo()+
         QString("<hr width=\"10000\"/>")+
         QString("<b>To-Point:</b> %1, %2, %3<br/>").arg(data[1]+toMove[0]).arg(data[2]+toMove[1]).arg(data[3]+toMove[2])+
         QString("<b>Vector:</b> %1, %2, %3<br/>").arg(data[4]*drFactor).arg(data[5]*drFactor).arg(data[6]*drFactor)+
         QString("<b>Length:</b> %1").arg(length/arrow->getScaleLength());
}

QMenu* Arrow::createMenu() {
  QMenu* menu=DynamicColoredBody::createMenu();
  menu->addSeparator()->setText("Properties from: Arrow");
  menu->addAction(path->getAction());
  menu->addAction(diameterEditor->getAction());
  menu->addAction(headDiameterEditor->getAction());
  menu->addAction(headLengthEditor->getAction());
  menu->addAction(scaleLengthEditor->getAction());
  return menu;
}
