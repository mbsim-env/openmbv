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

#ifndef _OPENMBVGUI_BODY_H_
#define _OPENMBVGUI_BODY_H_

#include "object.h"
#include <Inventor/C/errors/debugerror.h> // workaround a include order bug in Coin-3.1.3
#include <Inventor/sensors/SoFieldSensor.h>
#include <QActionGroup>
#include <Inventor/nodes/SoDrawStyle.h>
#include <Inventor/nodes/SoScale.h>
#include <Inventor/nodes/SoTriangleStripSet.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include "IndexedTesselationFace.h"
#include <Inventor/lists/SbVec3fList.h>
#include <Inventor/nodes/SoIndexedLineSet.h>
#include "utils.h"
#include "edgecalculation.h"
#include "editors.h"

namespace OpenMBV {
  class Body;
}

namespace OpenMBVGUI {

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
    Body(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind);
    ~Body() override;
    static void frameSensorCB(void *data, SoSensor*);
    virtual double update()=0; // return the current time
    void resetAnimRange(int numOfRows, double dt);
    static std::map<SoNode*,Body*>& getBodyMap() { return bodyMap; }
  protected:
    std::shared_ptr<OpenMBV::Body> body;
    SoSwitch *soOutLineSwitch, *soShilouetteEdgeSwitch;
    SoSeparator *soOutLineSep, *soShilouetteEdgeSep;
    static std::map<SoNode*,Body*> bodyMap;
    void createProperties() override;
    friend class IndexedTesselationFace;
    friend class MainWindow;
};

}

#endif
