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

#ifndef _OPENMBVGUI_RIGIDBODY_H_
#define _OPENMBVGUI_RIGIDBODY_H_

#include "dynamiccoloredbody.h"
#include <Inventor/C/errors/debugerror.h> // workaround a include order bug in Coin-3.1.3
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoRotationXYZ.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoRotation.h>
#include <hdf5serie/vectorserie.h>

namespace OpenMBV {
  class RigidBody;
}

namespace OpenMBVGUI {

class RigidBody : public DynamicColoredBody {
  Q_OBJECT
  friend class CompoundRigidBody;
  protected:
    std::shared_ptr<OpenMBV::RigidBody> rigidBody;
    SoSwitch *soLocalFrameSwitch, *soReferenceFrameSwitch, *soPathSwitch;
    SoCoordinate3 *pathCoord;
    SoLineSet *pathLine;
    int pathMaxFrameRead;
    double update() override;
    SoRotationXYZ *rotationAlpha, *rotationBeta, *rotationGamma;
    SoRotation *rotation; // accumulated rotationAlpha, rotationBeta and rotationGamma
    SoTranslation *translation;
    SoScale *refFrameScale, *localFrameScale, *scale;
    SoSeparator *soSepRigidBody;
    TransRotEditor *initialTransRotEditor;
    SoGroup *initTransRotGroup;
    void createProperties() override;
  public:
    RigidBody(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem_, SoGroup *soParent, int ind);
    ~RigidBody() override;
    QString getInfo() override;
    void moveCameraWithSlot();
};

}

#endif
