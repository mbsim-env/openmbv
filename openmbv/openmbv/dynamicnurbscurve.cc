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

DynamicNurbsCurve::DynamicNurbsCurve(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : DynamicColoredBody(obj, parentItem, soParent, ind) {
  nurbscurve=std::static_pointer_cast<OpenMBV::DynamicNurbsCurve>(obj);
  iconFile="nurbscurve.svg";
  setIcon(0, Utils::QIconCached(iconFile));

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
  int frame = MainWindow::getInstance()->getFrame()->getValue();
  std::vector<double> data = nurbscurve->getRow(frame);

  points->point.setNum(nurbscurve->getNumberOfControlPoints());
  SbVec4f *pointData = points->point.startEditing();
  for (int i=0; i<nurbscurve->getNumberOfControlPoints(); i++) {
    pointData[i][0] = data[i*4+1];
    pointData[i][1] = data[i*4+2];
    pointData[i][2] = data[i*4+3];
    pointData[i][3] = data[i*4+4];
  }
  points->point.finishEditing();
  points->point.setDefault(FALSE);
  return data[0]; //return time
}

}
