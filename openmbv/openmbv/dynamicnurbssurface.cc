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
#include "dynamicnurbssurface.h"
#include "utils.h"
#include "mainwindow.h"
#include <Inventor/nodes/SoNurbsSurface.h>
#include <Inventor/nodes/SoComplexity.h>
#include <Inventor/nodes/SoCoordinate4.h>
#include "openmbvcppinterface/dynamicnurbssurface.h"
#include <QMenu>
#include <vector>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

DynamicNurbsSurface::DynamicNurbsSurface(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : DynamicColoredBody(obj, parentItem, soParent, ind) {
  nurbssurface=std::static_pointer_cast<OpenMBV::DynamicNurbsSurface>(obj);
  iconFile="nurbssurface.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  vector<vector<double> > cp = nurbssurface->getControlPoints();

  points = new SoCoordinate4;

  vector<double> uKnot = nurbssurface->getUKnotVector();
  float u[uKnot.size()];
  for(unsigned int i=0; i<uKnot.size(); i++)
      u[i] = uKnot[i];

  vector<double> vKnot = nurbssurface->getVKnotVector();
  float v[vKnot.size()];
  for(unsigned int i=0; i<vKnot.size(); i++)
      v[i] = vKnot[i];

  auto *surface = new SoNurbsSurface;
  surface->numUControlPoints.setValue(nurbssurface->getNumberOfUControlPoints());
  surface->numVControlPoints.setValue(nurbssurface->getNumberOfVControlPoints());
  surface->uKnotVector.setValues(0, uKnot.size(), u);
  surface->vKnotVector.setValues(0, vKnot.size(), v);
  soSep->addChild(points);
  soSep->addChild(surface);

  // outline
  soSep->addChild(soOutLineSwitch);
}

double DynamicNurbsSurface::update() {
  int frame = MainWindow::getInstance()->getFrame()->getValue();
  std::vector<double> data = nurbssurface->getRow(frame);

  points->point.setNum(nurbssurface->getNumberOfUControlPoints()*nurbssurface->getNumberOfVControlPoints());
  SbVec4f *pointData = points->point.startEditing();
  for (int i=0; i<nurbssurface->getNumberOfUControlPoints()*nurbssurface->getNumberOfVControlPoints(); i++) {
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
