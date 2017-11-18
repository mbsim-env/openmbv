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

#ifndef _OPENMBVGUI_COILSPRING_H_
#define _OPENMBVGUI_COILSPRING_H_

#include "dynamiccoloredbody.h"
#include <Inventor/C/errors/debugerror.h> // workaround a include order bug in Coin-3.1.3
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/VRMLnodes/SoVRMLExtrusion.h>
#include <Inventor/nodes/SoRotation.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/SbLinear.h>
#include <hdf5serie/vectorserie.h>
#include <QtGui/QMenu>

namespace OpenMBV {
  class CoilSpring;
}

namespace OpenMBVGUI {

/**
 * \brief class for drawing simple helix springs
 * \author Thorsten Schindler
 * \date 2009-05-08 initial commit (Thorsten Schindler)
 * \date 2009-05-12 efficient spine update (Thorsten Schindler)
 * \todo setValuesPointer TODO
 */
class CoilSpring : public DynamicColoredBody {
  Q_OBJECT
  public:
    /** constructor */
    CoilSpring(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind);
    ~CoilSpring() override;

    /** info string in spine extrusion pop-up menu */
    QString getInfo() override;

  protected:

    /** extrusion body */
    SoVRMLExtrusion *extrusion;

    SoScale *scale;

    /** translation of helix */
    SoTranslation* fromPoint;

    /** rotation of helix */
    SoRotation* rotation;

    /** memory for efficient spine update */ 
    float* spine, *scaledSpine;

    /** radius of helix */
    double springRadius;

    /** mumber of coils */
    int numberOfCoils;

    /** number of spine points */
    static const int numberOfSpinePointsPerCoil = 120;

    /** cross section resolution */
    static const int iCircSegments = 20;

    /** scale factor */
    double scaleValue;

    double nominalLength, N;

    /** update method invoked at each time step */
    double update() override;

    std::shared_ptr<OpenMBV::CoilSpring> coilSpring;
    void createProperties() override;
};

}

#endif /* _COILSPRING_H_ */

