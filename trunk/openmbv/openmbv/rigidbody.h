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

#ifndef _MBSIMGUI_RIGIDBODY_H_
#define _MBSIMGUI_RIGIDBODY_H_

#include "dynamiccoloredbody.h"
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoRotationXYZ.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoRotation.h>
#include <H5Cpp.h>
#include <hdf5serie/vectorserie.h>

namespace OpenMBV {
  class RigidBody;
}

namespace OpenMBVGUI {

class RigidBody : public DynamicColoredBody {
  Q_OBJECT
  protected:
    OpenMBV::RigidBody *rigidBody;
    QAction *moveCameraWith;
    SoSwitch *soLocalFrameSwitch, *soReferenceFrameSwitch, *soPathSwitch;
    SoCoordinate3 *pathCoord;
    SoLineSet *pathLine;
    int pathMaxFrameRead;
    virtual double update();
    SoRotationXYZ *rotationAlpha, *rotationBeta, *rotationGamma;
    SoRotation *rotation; // accumulated rotationAlpha, rotationBeta and rotationGamma
    SoTranslation *translation;
    SoMaterial *mat;
    SoScale *refFrameScale, *localFrameScale;
    SoSeparator *soSepRigidBody;
    TransRotEditor *initialTransRotEditor;
  public:
    RigidBody(OpenMBV::Object* obj, QTreeWidgetItem *parentItem_, SoGroup *soParent, int ind);
    ~RigidBody();
    virtual QString getInfo();
  public slots:
    void moveCameraWithSlot();
};

}

#endif
