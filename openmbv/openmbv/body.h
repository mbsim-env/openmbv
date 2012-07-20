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

#ifndef _BODY_H_
#define _BODY_H_

#include "config.h"
#include "object.h"
#include <Inventor/sensors/SoFieldSensor.h>
#include <H5Cpp.h>
#include <QtGui/QActionGroup>
#include <Inventor/nodes/SoDrawStyle.h>
#include <Inventor/nodes/SoScale.h>
#include <Inventor/nodes/SoTriangleStripSet.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include "IndexedTesselationFace.h"
#include <Inventor/lists/SbVec3fList.h>
#include <Inventor/nodes/SoIndexedLineSet.h>
#include "utils.h"
#include "edgecalculation.h"
#include <editors.h>

namespace OpenMBV {
  class Body;
}

class Body : public Object {
  Q_OBJECT
  private:
    SoDrawStyle *drawStyle;
    SoFieldSensor *shilouetteEdgeFrameSensor, *shilouetteEdgeOrientationSensor;
    static void shilouetteEdgeFrameOrCameraSensorCB(void *data, SoSensor* sensor);
    // for shilouetteEdge
    SoCoordinate3 *soShilouetteEdgeCoord;
    SoIndexedLineSet *soShilouetteEdge;
    bool shilouetteEdgeFirstCall;
    EdgeCalculation *edgeCalc;
    SoFieldSensor *frameSensor;
  public:
    Body(OpenMBV::Object* obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind);
    ~Body();
    static void frameSensorCB(void *data, SoSensor*);
    virtual double update()=0; // return the current time
    void resetAnimRange(int numOfRows, double dt);
  public slots:
    static std::map<SoNode*,Body*>& getBodyMap() { return bodyMap; }
  protected:
    OpenMBV::Body *body;
    SoSwitch *soOutLineSwitch, *soShilouetteEdgeSwitch;
    SoSeparator *soOutLineSep, *soShilouetteEdgeSep;
    static std::map<SoNode*,Body*> bodyMap;
    friend class IndexedTesselationFace;
    friend class MainWindow;
};

#endif
