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
#include <Inventor/nodes/SoDrawStyle.h>
#include <Inventor/nodes/SoShapeHints.h>
#include "utils.h"
#include "openmbvcppinterface/arrow.h"
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

Arrow::Arrow(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : DynamicColoredBody(obj, parentItem, soParent, ind) {
  arrow=std::static_pointer_cast<OpenMBV::Arrow>(obj);
  iconFile="arrow.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  //h5 dataset
  int rows=arrow->getRows();
  double dt;
  if(rows>=2) dt=arrow->getRow(1)[0]-arrow->getRow(0)[0]; else dt=0;
  resetAnimRange(rows, dt);

  // read XML
  headLength=arrow->getHeadLength();
  scaleLength=arrow->getScaleLength();
  double headDiameter=arrow->getHeadDiameter();
  double diameter=arrow->getDiameter();

  // create so
  // path
  soPathSwitch=new SoSwitch;
  soSep->addChild(soPathSwitch);
  auto *pathSep=new SoSeparator;
  soPathSwitch->addChild(pathSep);
  auto *col=new SoBaseColor;
  col->rgb.setValue(0, 1, 0);
  pathSep->addChild(col);
  pathCoord=new SoCoordinate3;
  pathCoord->point.setNum(0);
  pathSep->addChild(pathCoord);
  pathLine=new SoLineSet;
  pathLine->numVertices.setNum(0);
  pathSep->addChild(pathLine);
  soPathSwitch->whichChild.setValue(arrow->getPath()?SO_SWITCH_ALL:SO_SWITCH_NONE);
  pathMaxFrameRead=-1;
  pathNewLine=true;

  // separator for the arrow
  soArrowSwitch=new SoSwitch;
  soSep->addChild(soArrowSwitch);
  soArrowSwitch->whichChild.setValue(SO_SWITCH_ALL);

  if(arrow->getType()==OpenMBV::Arrow::line) {
    // Line
 
    // line width
    auto *drawStyle=new SoDrawStyle;
    soArrowSwitch->addChild(drawStyle);
    drawStyle->lineWidth.setValue(2);
    // line
    lineCoord=new SoCoordinate3;
    soArrowSwitch->addChild(lineCoord);
    auto *ls=new SoLineSet;
    soArrowSwitch->addChild(ls);
    ls->numVertices.set1Value(0, 2);
  }
  else {
    // Arrow
    // translate to To-Point
    toPoint=new SoTranslation;
    soArrowSwitch->addChild(toPoint);
    // rotation to dx,dy,dz
    rotation1=new SoRotation;
    soArrowSwitch->addChild(rotation1);
    rotation2=new SoRotation;
    soArrowSwitch->addChild(rotation2);
    // arrow separator
    auto *arrowSep=new SoSeparator;
    soArrowSwitch->addChild(arrowSep);
    // add arrow twice for type==bothHeads with half length: mirrored by x-z-plane and moved
    if(arrow->getType()==OpenMBV::Arrow::bothHeads || arrow->getType()==OpenMBV::Arrow::bothDoubleHeads) {
      auto *bScale=new SoScale;
      soArrowSwitch->addChild(bScale);
      // scale y by -1 to get the mirrored arrow and scale x by -1 to get the same vertex ordering without changing the geometry
      bScale->scaleFactor.setValue(-1,-1,1);
      bTrans=new SoTranslation;
      soArrowSwitch->addChild(bTrans);
      soArrowSwitch->addChild(arrowSep);
    }
    // full scale (l<2*headLength)
    scale1=new SoScale;
    arrowSep->addChild(scale1);
    // outline
    arrowSep->addChild(soOutLineSwitch);
    // trans1
    auto *trans1=new SoTranslation;
    arrowSep->addChild(trans1);
    trans1->translation.setValue(0, -headLength/2, 0);
    // cone
    auto *cone1=new SoCone;
    arrowSep->addChild(cone1);
    cone1->bottomRadius.setValue(headDiameter/2);
    cone1->height.setValue(headLength);
    // add the head twice for double heads
    SoTranslation *dTrans=nullptr;
    double dTranslate=0;
    if(arrow->getType()==OpenMBV::Arrow::fromDoubleHead ||
       arrow->getType()==OpenMBV::Arrow::toDoubleHead ||
       arrow->getType()==OpenMBV::Arrow::bothDoubleHeads) {
      dTrans=new SoTranslation;
      arrowSep->addChild(dTrans);
      dTranslate=headLength*(headDiameter-diameter)/headDiameter;
      dTrans->translation.setValue(0, -dTranslate, 0);
      arrowSep->addChild(cone1);
      dTrans=new SoTranslation;
      arrowSep->addChild(dTrans);
      dTrans->translation.setValue(0, dTranslate, 0);
    }
    // trans2
    auto *trans2=new SoTranslation;
    arrowSep->addChild(trans2);
    trans2->translation.setValue(0, -headLength/2, 0);
    // scale only arrow not head (l>2*headLength)
    scale2=new SoScale;
    arrowSep->addChild(scale2);
    // trans2
    auto *trans3=new SoTranslation;
    arrowSep->addChild(trans3);
    trans3->translation.setValue(0, -headLength/2, 0);
    // cylinder
    auto *cylinder=new SoCylinder;
    arrowSep->addChild(cylinder);
    cylinder->radius.setValue(diameter/2);
    cylinder->height.setValue(headLength);

    // outline
    auto *transLO1=new SoTranslation;
    soOutLineSep->addChild(transLO1);
    transLO1->translation.setValue(0, -headLength, 0);
    auto *cylOL1=new SoCylinder;
    soOutLineSep->addChild(cylOL1);
    cylOL1->height.setValue(0);
    cylOL1->radius.setValue(headDiameter/2);
    cylOL1->parts.setValue(SoCylinder::SIDES);
    auto *cylOL2=new SoCylinder;
    soOutLineSep->addChild(cylOL2);
    cylOL2->height.setValue(0);
    cylOL2->radius.setValue(diameter/2);
    cylOL2->parts.setValue(SoCylinder::SIDES);
    // add the head outline twice for double heads
    if(arrow->getType()==OpenMBV::Arrow::fromDoubleHead ||
       arrow->getType()==OpenMBV::Arrow::toDoubleHead ||
       arrow->getType()==OpenMBV::Arrow::bothDoubleHeads) {
      dTrans=new SoTranslation;
      soOutLineSep->addChild(dTrans);
      dTrans->translation.setValue(0, -dTranslate, 0);
      soOutLineSep->addChild(cylOL1);
      soOutLineSep->addChild(cylOL2);
      dTrans=new SoTranslation;
      soOutLineSep->addChild(dTrans);
      dTrans->translation.setValue(0, dTranslate, 0);
    }
    // for type==bothHeads do not draw the middle outline circle
    if(arrow->getType()!=OpenMBV::Arrow::bothHeads && arrow->getType()!=OpenMBV::Arrow::bothDoubleHeads) {
      soOutLineSep->addChild(scale2);
      auto *transLO2=new SoTranslation;
      soOutLineSep->addChild(transLO2);
      transLO2->translation.setValue(0, -headLength, 0);
      soOutLineSep->addChild(cylOL2);
    }
  }
}
 
void Arrow::createProperties() {
  DynamicColoredBody::createProperties();

  // GUI
  if(!clone) {
    properties->updateHeader();
    BoolEditor *pathEditor=new BoolEditor(properties, Utils::QIconCached("path.svg"),"Draw path of to-point", "Arrow::path");
    pathEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getPath, &OpenMBV::Arrow::setPath);
    properties->addPropertyAction(pathEditor->getAction());

    FloatEditor *diameterEditor=new FloatEditor(properties, QIcon(), "Diameter");
    diameterEditor->setRange(0, DBL_MAX);
    diameterEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getDiameter, &OpenMBV::Arrow::setDiameter);

    FloatEditor *headDiameterEditor=new FloatEditor(properties, QIcon(), "Head diameter");
    headDiameterEditor->setRange(0, DBL_MAX);
    headDiameterEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getHeadDiameter, &OpenMBV::Arrow::setHeadDiameter);

    FloatEditor *headLengthEditor=new FloatEditor(properties, QIcon(), "Head length");
    headLengthEditor->setRange(0, DBL_MAX);
    headLengthEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getHeadLength, &OpenMBV::Arrow::setHeadLength);

    ComboBoxEditor *typeEditor=new ComboBoxEditor(properties, QIcon(), "Type", {
      make_tuple(OpenMBV::Arrow::line,            "Line",              QIcon(), "Arrow::type::line"),
      make_tuple(OpenMBV::Arrow::fromHead,        "From head",         QIcon(), "Arrow::type::fromHead"),
      make_tuple(OpenMBV::Arrow::toHead,          "To head",           QIcon(), "Arrow::type::toHead"),
      make_tuple(OpenMBV::Arrow::bothHeads,       "Both heads",        QIcon(), "Arrow::type::bothHeads"),
      make_tuple(OpenMBV::Arrow::fromDoubleHead,  "From double head",  QIcon(), "Arrow::type::fromDoubleHead"),
      make_tuple(OpenMBV::Arrow::toDoubleHead,    "To double head",    QIcon(), "Arrow::type::toDoubleHead"),
      make_tuple(OpenMBV::Arrow::bothDoubleHeads, "Both double heads", QIcon(), "Arrow::type::bothDoubleHeads")
    });
    typeEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getType, &OpenMBV::Arrow::setType);

    ComboBoxEditor *referencePointEditor=new ComboBoxEditor(properties, QIcon(), "Reference Point", {
      make_tuple(OpenMBV::Arrow::toPoint,   "To point",   QIcon(), "Arrow::referencePoint::toPoint"),
      make_tuple(OpenMBV::Arrow::fromPoint, "From point", QIcon(), "Arrow::referencePoint::fromPoint"),
      make_tuple(OpenMBV::Arrow::midPoint,  "Mid point",  QIcon(), "Arrow::referencePoint::midPoint")
    });
    referencePointEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getReferencePoint, &OpenMBV::Arrow::setReferencePoint);

    FloatEditor *scaleLengthEditor=new FloatEditor(properties, QIcon(), "Scale length");
    scaleLengthEditor->setRange(0, DBL_MAX);
    scaleLengthEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getScaleLength, &OpenMBV::Arrow::setScaleLength);
  }
}

double Arrow::update() {
  int frame=MainWindow::getInstance()->getFrame()->getValue();
  // read from hdf5
  data=arrow->getRow(frame);

  // convert data from referencePoint to toPoint reference
  if(arrow->getReferencePoint()==OpenMBV::Arrow::fromPoint) {
    data[1]+=data[4]*arrow->getScaleLength();
    data[2]+=data[5]*arrow->getScaleLength();
    data[3]+=data[6]*arrow->getScaleLength();
  }
  else if(arrow->getReferencePoint()==OpenMBV::Arrow::midPoint) {
    data[1]+=data[4]*arrow->getScaleLength()/2;
    data[2]+=data[5]*arrow->getScaleLength()/2;
    data[3]+=data[6]*arrow->getScaleLength()/2;
  }
 
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
  length=sqrt(dx*dx+dy*dy+dz*dz)*scaleLength;

  // path
  if(arrow->getPath()) {
    for(int i=pathMaxFrameRead+1; i<=frame; i++) {
      vector<double> localData=arrow->getRow(i);
      if(localData[4]*localData[4]+localData[5]*localData[5]+localData[6]*localData[6]<1e-10) {
        pathNewLine=true;
        continue;
      }
      if(pathNewLine)
        pathLine->numVertices.set1Value(pathLine->numVertices.getNum(), 0);
      pathLine->numVertices.set1Value(pathLine->numVertices.getNum()-1, pathLine->numVertices[pathLine->numVertices.getNum()-1]+1);
      pathCoord->point.set1Value(pathCoord->point.getNum(), localData[1], localData[2], localData[3]);
      pathNewLine=false;
    }
    pathMaxFrameRead=max(pathMaxFrameRead, frame);
  }

  // if length is 0 do not draw
  if(length<1e-10) {
    soArrowSwitch->whichChild.setValue(SO_SWITCH_NONE);
    soBBoxSwitch->whichChild.setValue(SO_SWITCH_NONE);
    return data[0];
  }
  else {
    soArrowSwitch->whichChild.setValue(SO_SWITCH_ALL);
    if(object->getBoundingBox()) soBBoxSwitch->whichChild.setValue(SO_SWITCH_ALL);
  }

  // if type==line: draw update line and exit
  if(arrow->getType()==OpenMBV::Arrow::line) {
    if(diffuseColor[0]<0) setColor(data[7]);
    lineCoord->point.set1Value(0, data[1], data[2], data[3]); // to point
    lineCoord->point.set1Value(1, data[1]-dx*scaleLength, data[2]-dy*scaleLength, data[3]-dz*scaleLength); // from point
    return data[0];
  }

  // type!=line: draw arrow

  // for type==bothHeads: set translation of second arrow and draw two arrows with half length
  if(arrow->getType()==OpenMBV::Arrow::bothHeads || arrow->getType()==OpenMBV::Arrow::bothDoubleHeads) {
    bTrans->translation.setValue(0, length, 0);
    length/=2.0;
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
  if(diffuseColor[0]<0) setColor(data[7]);

  // for type==bothHeads: reset half length
  if(arrow->getType()==OpenMBV::Arrow::bothHeads || arrow->getType()==OpenMBV::Arrow::bothDoubleHeads)
    length*=2.0;

  return data[0];
}

QString Arrow::getInfo() {
  if(data.empty()) update();

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
         QString("<b>To-point:</b> %1, %2, %3<br/>").arg(data[1]+toMove[0]).arg(data[2]+toMove[1]).arg(data[3]+toMove[2])+
         QString("<b>Vector:</b> %1, %2, %3<br/>").arg(data[4]*drFactor).arg(data[5]*drFactor).arg(data[6]*drFactor)+
         QString("<b>Length:</b> %1").arg(length/scaleLength);
}

}
