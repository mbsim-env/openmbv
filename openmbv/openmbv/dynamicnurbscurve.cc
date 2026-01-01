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
#include "dynamicnurbscurve.h"
#include "utils.h"
#include "mainwindow.h"
#include <Inventor/nodes/SoNurbsCurve.h>
#include <Inventor/nodes/SoComplexity.h>
#include <Inventor/nodes/SoCoordinate4.h>
#include "openmbvcppinterface/dynamicnurbscurve.h"
#include <QMenu>
#include <vector>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

DynamicNurbsCurve::DynamicNurbsCurve(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : DynamicColoredBody(obj, parentItem, soParent, ind, true) {
  nurbscurve=std::static_pointer_cast<OpenMBV::DynamicNurbsCurve>(obj);
  //iconFile="nurbscurve.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  mat->diffuseColor.setNum(nurbscurve->getNumberOfControlPoints());
  mat->specularColor.setNum(nurbscurve->getNumberOfControlPoints());

  vector<vector<double> > cp = nurbscurve->getControlPoints();

  points = new SoCoordinate4;

  vector<double> knot = nurbscurve->getKnotVector();
  float u[knot.size()];
  for(unsigned int i=0; i<knot.size(); i++)
    u[i] = knot[i];

  auto *curve = new SoNurbsCurve;
  curve->numControlPoints.setValue(nurbscurve->getNumberOfControlPoints());
  curve->knotVector.setValues(0, knot.size(), u);
  soSep->addChild(points);
  soSep->addChild(curve);

  // outline
  soSep->addChild(soOutLineSwitch);
}

double DynamicNurbsCurve::update() {
  int frame = MainWindow::getInstance()->getFrame()[0];
  std::vector<double> data = nurbscurve->getRow(frame);

  SbColor *colorData = mat->diffuseColor.startEditing();
  SbColor *specData = mat->specularColor.startEditing();
  float h, s, v;
  mat->diffuseColor[0].getHSVValue(h, s, v);
  double m=1/(maximalColorValue-minimalColorValue);

  points->point.setNum(nurbscurve->getNumberOfControlPoints());
  SbVec4f *pointData = points->point.startEditing();
  for (int i=0; i<nurbscurve->getNumberOfControlPoints(); i++) {
    double hue = diffuseColor[0];
    if(hue<0) {
    double col = data[i*5+5];
    col = m*(col-minimalColorValue);
    if(col<0) col=0;
    if(col>1) col=1;
    hue = (1-col)*2/3;
    }
    colorData[i].setHSVValue(hue, s, v);
    specData[i].setHSVValue(hue, 0.7*s, v);
    pointData[i][0] = data[i*5+1];
    pointData[i][1] = data[i*5+2];
    pointData[i][2] = data[i*5+3];
    pointData[i][3] = data[i*5+4];
  }
  mat->diffuseColor.finishEditing();
  mat->diffuseColor.setDefault(FALSE);
  mat->specularColor.finishEditing();
  mat->specularColor.setDefault(FALSE);
  points->point.finishEditing();
  points->point.setDefault(FALSE);
  return data[0]; //return time
}

}
