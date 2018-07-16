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
#include "flexiblebody.h"
#include "utils.h"
#include "mainwindow.h"
#include <Inventor/nodes/SoComplexity.h>
#include "openmbvcppinterface/flexiblebody.h"
#include <QMenu>
#include <vector>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

FlexibleBody::FlexibleBody(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : DynamicColoredBody(obj, parentItem, soParent, ind, true) {
  body=std::static_pointer_cast<OpenMBV::FlexibleBody>(obj);
  iconFile="flexiblebody.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  mat->diffuseColor.setNum(body->getNumberOfVertexPositions());
  mat->specularColor.setNum(body->getNumberOfVertexPositions());

  points = new SoCoordinate3;
  soSep->addChild(points);

  // outline
  soSep->addChild(soOutLineSwitch);
}

}
