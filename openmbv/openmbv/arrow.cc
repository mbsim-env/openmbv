/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

  This library is free software; you can redistribute it and/or 
  modify it under the terms of the GNU Lesser General Public 
  License as published by the Free Software Foundation; either 
  version 2.1 of the License, or (at your option) any later version. 
   
  This library is distributed in the hope that it will be useful, 
  but WITHOUT ANY WARRANTY; without even the implied warranty of 
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
  Lesser General Public License for more details. 
   
  You should have received a copy of the GNU Lesser General Public 
  License along with this library; if not, write to the Free Software 
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
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

  components = arrow->getComponents();
  if(components==OpenMBV::Arrow::vectorForm)
    ctorComponents(0);
  else {
    ctorComponents(0);
    ctorComponents(1);
    ctorComponents(2);
  }
}

void Arrow::ctorComponents(int c) {
  double headDiameter=arrow->getHeadDiameter();
  double diameter=arrow->getDiameter();

  // separator for the arrow
  auto soArrowSep=new SoSeparator;
  soSep->addChild(soArrowSep);
  soArrowSwitch[c]=new SoSwitch;
  soArrowSep->addChild(soArrowSwitch[c]);
  soArrowSwitch[c]->whichChild.setValue(SO_SWITCH_ALL);

  if(arrow->getType()==OpenMBV::Arrow::line) {
    // Line
 
    // line
    lineCoord[c]=new SoCoordinate3;
    soArrowSwitch[c]->addChild(lineCoord[c]);
    auto *ls=new SoLineSet;
    soArrowSwitch[c]->addChild(ls);
    ls->numVertices.set1Value(0, 2);
  }
  else {
    // Arrow
    // translate to To-Point
    toPoint[c]=new SoTranslation;
    soArrowSwitch[c]->addChild(toPoint[c]);
    // rotation to dx,dy,dz
    rotation1[c]=new SoRotation;
    soArrowSwitch[c]->addChild(rotation1[c]);
    rotation2[c]=new SoRotation;
    soArrowSwitch[c]->addChild(rotation2[c]);
    // arrow separator
    auto *arrowSep=new SoSeparator;
    soArrowSwitch[c]->addChild(arrowSep);
    // add arrow twice for type==bothHeads with half length: mirrored by x-z-plane and moved
    if(arrow->getType()==OpenMBV::Arrow::bothHeads || arrow->getType()==OpenMBV::Arrow::bothDoubleHeads) {
      auto *bScale=new SoScale;
      soArrowSwitch[c]->addChild(bScale);
      // scale y by -1 to get the mirrored arrow and scale x by -1 to get the same vertex ordering without changing the geometry
      bScale->scaleFactor.setValue(-1,-1,1);
      bTrans[c]=new SoTranslation;
      soArrowSwitch[c]->addChild(bTrans[c]);
      soArrowSwitch[c]->addChild(arrowSep);
    }
    // full scale (l<2*headLength)
    scale1[c]=new SoScale;
    arrowSep->addChild(scale1[c]);
    // outline
    soOutLineSwitch[c] = new SoSwitch;
    soOutLineSwitch[c]->whichChild.connectFrom(&DynamicColoredBody::soOutLineSwitch->whichChild);
    soOutLineSep[c] = new SoSeparator;
    soOutLineSwitch[c]->addChild(soOutLineSep[c]);
    soOutLineSep[c]->addChild(DynamicColoredBody::soOutLineSep->getChild(0));
    arrowSep->addChild(soOutLineSwitch[c]);
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
    scale2[c]=new SoScale;
    arrowSep->addChild(scale2[c]);
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
    auto *olSep=new SoSeparator;
    soOutLineSep[c]->addChild(olSep);
    auto *transLO1=new SoTranslation;
    olSep->addChild(transLO1);
    transLO1->translation.setValue(0, -headLength, 0);

    auto outlineCircle = [](double diameter) {
      int num = 500 * appSettings->get<double>(AppSettings::complexityValue);
      vector<array<float,3>> pts(num);
      for(int i=0; i<num; ++i) {
        double angle = 2.0*M_PI*i/num;
        pts[i][0] = diameter/2*sin(angle);
        pts[i][1] = 0;
        pts[i][2] = diameter/2*cos(angle);
      }
      auto *points = new SoCoordinate3;
      points->point.setValues(0, num, reinterpret_cast<float(*)[3]>(pts.data()));
      points->point.setValues(num, 1, reinterpret_cast<float(*)[3]>(pts.data()));
      auto *line = new SoLineSet;
      line->numVertices.setValue(num+1);
      auto *cylOL = new SoSeparator;
      cylOL->addChild(points);
      cylOL->addChild(line);
      return cylOL;
    };
    auto cylOL1 = outlineCircle(headDiameter);
    olSep->addChild(cylOL1);
    auto *cylOL2 = outlineCircle(diameter);
    olSep->addChild(cylOL2);
    // add the head outline twice for double heads
    if(arrow->getType()==OpenMBV::Arrow::fromDoubleHead ||
       arrow->getType()==OpenMBV::Arrow::toDoubleHead ||
       arrow->getType()==OpenMBV::Arrow::bothDoubleHeads) {
      dTrans=new SoTranslation;
      olSep->addChild(dTrans);
      dTrans->translation.setValue(0, -dTranslate, 0);
      olSep->addChild(cylOL1);
      olSep->addChild(cylOL2);
      dTrans=new SoTranslation;
      olSep->addChild(dTrans);
      dTrans->translation.setValue(0, dTranslate, 0);
    }
    // for type==bothHeads do not draw the middle outline circle
    if(arrow->getType()!=OpenMBV::Arrow::bothHeads && arrow->getType()!=OpenMBV::Arrow::bothDoubleHeads) {
      olSep->addChild(scale2[c]);
      auto *transLO2=new SoTranslation;
      olSep->addChild(transLO2);
      transLO2->translation.setValue(0, -headLength, 0);
      olSep->addChild(cylOL2);
    }
  }
}

void Arrow::createProperties() {
  DynamicColoredBody::createProperties();

  // GUI
  if(!clone) {
    properties->updateHeader();
    auto *pathEditor=new BoolEditor(properties, Utils::QIconCached("path.svg"),"Draw path of to-point", "Arrow::path");
    pathEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getPath, &OpenMBV::Arrow::setPath);
    properties->addPropertyAction(pathEditor->getAction());

    auto *diameterEditor=new FloatEditor(properties, QIcon(), "Diameter");
    diameterEditor->setRange(0, DBL_MAX);
    diameterEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getDiameter, &OpenMBV::Arrow::setDiameter);

    auto *headDiameterEditor=new FloatEditor(properties, QIcon(), "Head diameter");
    headDiameterEditor->setRange(0, DBL_MAX);
    headDiameterEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getHeadDiameter, &OpenMBV::Arrow::setHeadDiameter);

    auto *headLengthEditor=new FloatEditor(properties, QIcon(), "Head length");
    headLengthEditor->setRange(0, DBL_MAX);
    headLengthEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getHeadLength, &OpenMBV::Arrow::setHeadLength);

    auto *typeEditor=new ComboBoxEditor(properties, QIcon(), "Type", {
      make_tuple(OpenMBV::Arrow::line,            "Line",              QIcon(), "Arrow::type::line"),
      make_tuple(OpenMBV::Arrow::fromHead,        "From head",         QIcon(), "Arrow::type::fromHead"),
      make_tuple(OpenMBV::Arrow::toHead,          "To head",           QIcon(), "Arrow::type::toHead"),
      make_tuple(OpenMBV::Arrow::bothHeads,       "Both heads",        QIcon(), "Arrow::type::bothHeads"),
      make_tuple(OpenMBV::Arrow::fromDoubleHead,  "From double head",  QIcon(), "Arrow::type::fromDoubleHead"),
      make_tuple(OpenMBV::Arrow::toDoubleHead,    "To double head",    QIcon(), "Arrow::type::toDoubleHead"),
      make_tuple(OpenMBV::Arrow::bothDoubleHeads, "Both double heads", QIcon(), "Arrow::type::bothDoubleHeads")
    });
    typeEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getType, &OpenMBV::Arrow::setType);

    vector<tuple<int, string, QIcon, string>> list {
      make_tuple(OpenMBV::Arrow::vectorForm,        "Vector form",         QIcon(), "Arrow::components::vectorForm"),
      make_tuple(OpenMBV::Arrow::componentsInWorld, "Components in World", QIcon(), "Arrow::components::componentsInWorld"),
    };
    if(arrow->getRow(0).size()==11)
      list.emplace_back(OpenMBV::Arrow::componentsInLocal, "Components in Local", QIcon(), "Arrow::components::componentsInLocal");
    auto *componentsEditor=new ComboBoxEditor(properties, QIcon(), "Components", list);
    componentsEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getComponents, &OpenMBV::Arrow::setComponents);

    auto *referencePointEditor=new ComboBoxEditor(properties, QIcon(), "Reference Point", {
      make_tuple(OpenMBV::Arrow::toPoint,   "To point",   QIcon(), "Arrow::referencePoint::toPoint"),
      make_tuple(OpenMBV::Arrow::fromPoint, "From point", QIcon(), "Arrow::referencePoint::fromPoint"),
      make_tuple(OpenMBV::Arrow::midPoint,  "Mid point",  QIcon(), "Arrow::referencePoint::midPoint")
    });
    referencePointEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getReferencePoint, &OpenMBV::Arrow::setReferencePoint);

    auto *scaleLengthEditor=new FloatEditor(properties, QIcon(), "Scale length");
    scaleLengthEditor->setOpenMBVParameter(arrow, &OpenMBV::Arrow::getScaleLength, &OpenMBV::Arrow::setScaleLength);
  }
}

double Arrow::update() {
  // read from hdf5
  auto dataHDF5=arrow->getRow(MainWindow::getInstance()->getFrame()[0]);

  if(components==OpenMBV::Arrow::vectorForm)
    data[0] = dataHDF5;
  else if(components==OpenMBV::Arrow::componentsInWorld) {
    data[0] = dataHDF5;
    data[1] = dataHDF5;
    data[2] = dataHDF5;
                    data[0][5] = 0; data[0][6] = 0;
    data[1][4] = 0;                 data[1][6] = 0;
    data[2][4] = 0; data[2][5] = 0;
  }
  else if(components==OpenMBV::Arrow::componentsInLocal) {
    data[0] = dataHDF5;
    data[1] = dataHDF5;
    data[2] = dataHDF5;
    SbVec3f W_F(dataHDF5[4], dataHDF5[5], dataHDF5[6]);
    auto T_LW = Utils::cardan2Rotation(SbVec3f(dataHDF5[8], dataHDF5[9], dataHDF5[10]));
    SbVec3f L_F;
    T_LW.multVec(W_F, L_F);
    T_LW.invert(); // T_LW is now T_WL
    { SbVec3f x; T_LW.multVec(SbVec3f(L_F[0],0     ,0)     , x); x.getValue(data[0][4], data[0][5], data[0][6]); }
    { SbVec3f x; T_LW.multVec(SbVec3f(0     ,L_F[1],0)     , x); x.getValue(data[1][4], data[1][5], data[1][6]); }
    { SbVec3f x; T_LW.multVec(SbVec3f(0     ,0     ,L_F[2]), x); x.getValue(data[2][4], data[2][5], data[2][6]); }
  }

  // path
  int frame=MainWindow::getInstance()->getFrame()[0];
  if(arrow->getPath()) {
    for(int i=pathMaxFrameRead+1; i<=frame; i++) {
      auto localData=arrow->getRow(i);
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

  if(components==OpenMBV::Arrow::vectorForm) {
    return updateComponents(0);
  }
  else {
           updateComponents(0);
           updateComponents(1);
    return updateComponents(2);
  }
}

double Arrow::updateComponents(int c) {
  // convert data from referencePoint to toPoint reference
  if(arrow->getReferencePoint()==OpenMBV::Arrow::fromPoint) {
    data[c][1]+=data[c][4]*arrow->getScaleLength();
    data[c][2]+=data[c][5]*arrow->getScaleLength();
    data[c][3]+=data[c][6]*arrow->getScaleLength();
  }
  else if(arrow->getReferencePoint()==OpenMBV::Arrow::midPoint) {
    data[c][1]+=data[c][4]*arrow->getScaleLength()/2;
    data[c][2]+=data[c][5]*arrow->getScaleLength()/2;
    data[c][3]+=data[c][6]*arrow->getScaleLength()/2;
  }
 
  // convert data from fromHead representation to toHead representation
  if(arrow->getType()==OpenMBV::Arrow::fromHead || arrow->getType()==OpenMBV::Arrow::fromDoubleHead)
  {
    // to-point_new=to-point_old-dv_old; dv_new=-dv_old
    data[c][1]-=data[c][4]*arrow->getScaleLength();
    data[c][2]-=data[c][5]*arrow->getScaleLength();
    data[c][3]-=data[c][6]*arrow->getScaleLength();
    data[c][4]*=-1.0;
    data[c][5]*=-1.0;
    data[c][6]*=-1.0;
  }

  // set scene values
  double dx=data[c][4], dy=data[c][5], dz=data[c][6];
  // handle negative scaleLength
  double sl=scaleLength;
  if(scaleLength<0) {
    sl*=-1;
    dx*=-1;
    dy*=-1;
    dz*=-1;
  }
  length[c]=sqrt(dx*dx+dy*dy+dz*dz)*sl;

  // if length is 0 do not draw
  if(length[c]<1e-10) {
    soArrowSwitch[c]->whichChild.setValue(SO_SWITCH_NONE);
    soBBoxSwitch->whichChild.setValue(SO_SWITCH_NONE);
    return data[c][0];
  }
  else {
    soArrowSwitch[c]->whichChild.setValue(SO_SWITCH_ALL);
    if(drawBoundingBox()) soBBoxSwitch->whichChild.setValue(SO_SWITCH_ALL);
  }

  // if type==line: draw update line and exit
  if(arrow->getType()==OpenMBV::Arrow::line) {
    if(diffuseColor[0]<0) setColor(data[c][7]);
    lineCoord[c]->point.set1Value(0, data[c][1], data[c][2], data[c][3]); // to point
    lineCoord[c]->point.set1Value(1, data[c][1]-dx*sl, data[c][2]-dy*sl, data[c][3]-dz*sl); // from point
    return data[c][0];
  }

  // type!=line: draw arrow

  // for type==bothHeads: set translation of second arrow and draw two arrows with half length
  if(arrow->getType()==OpenMBV::Arrow::bothHeads || arrow->getType()==OpenMBV::Arrow::bothDoubleHeads) {
    bTrans[c]->translation.setValue(0, length[c], 0);
    length[c]/=2.0;
  }

  // scale factors
  if(length[c]<2*headLength)
    scale1[c]->scaleFactor.setValue(length[c]/2/headLength,length[c]/2/headLength,length[c]/2/headLength);
  else
    scale1[c]->scaleFactor.setValue(1,1,1);
  if(length[c]>2*headLength)
    scale2[c]->scaleFactor.setValue(1,(length[c]-headLength)/headLength,1);
  else
    scale2[c]->scaleFactor.setValue(1,1,1);
  // trans to To-Point
  toPoint[c]->translation.setValue(data[c][1], data[c][2], data[c][3]);
  // rotation to dx,dy,dz
  rotation1[c]->rotation.setValue(SbVec3f(0, 0, 1), -atan2(dx,dy));
  rotation2[c]->rotation.setValue(SbVec3f(1, 0, 0), atan2(dz,sqrt(dx*dx+dy*dy)));
  // mat
  if(diffuseColor[0]<0) setColor(data[c][7]);

  // for type==bothHeads: reset half length
  if(arrow->getType()==OpenMBV::Arrow::bothHeads || arrow->getType()==OpenMBV::Arrow::bothDoubleHeads)
    length[c]*=2.0;

  return data[c][0];
}

QString Arrow::getInfo() {
  if(data[0].empty()) update();

  // convert data back from toHead to fromHead if type==fromHead or fromDoubleHead
  double drFactor=1;
  double toMove[]={0, 0, 0};
  if(arrow->getType()==OpenMBV::Arrow::fromHead || arrow->getType()==OpenMBV::Arrow::fromDoubleHead)
  {
    drFactor=-1;
    toMove[0]=-data[0][4]*arrow->getScaleLength();
    toMove[1]=-data[0][5]*arrow->getScaleLength();
    toMove[2]=-data[0][6]*arrow->getScaleLength();
  }

  QString ret(DynamicColoredBody::getInfo()+
              QString("<hr width=\"10000\"/>")+
              QString("<b>To-point:</b> %1, %2, %3<br/>").arg(data[0][1]+toMove[0]).arg(data[0][2]+toMove[1]).arg(data[0][3]+toMove[2])
             );
  if(components==OpenMBV::Arrow::vectorForm)
    ret+=QString("<b>Vector:</b> %1, %2, %3<br/>").arg(data[0][4]*drFactor).arg(data[0][5]*drFactor).arg(data[0][6]*drFactor)+
         QString("<b>Length:</b> %1").arg(length[0]/abs(scaleLength));
  else {
    QString frame(components==OpenMBV::Arrow::componentsInWorld ? "W" : "L");
    ret+=QString("<b>Length %2_x:</b> %1<br/>").arg(length[0]/abs(scaleLength)).arg(frame)+
         QString("<b>Length %2_y:</b> %1<br/>").arg(length[1]/abs(scaleLength)).arg(frame)+
         QString("<b>Length %2_z:</b> %1").arg(length[2]/abs(scaleLength)).arg(frame);
  }
  return ret;
}

}
