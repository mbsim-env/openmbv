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

#ifndef _COILSPRING_H_
#define _COILSPRING_H_

#include "config.h"
#include "dynamiccoloredbody.h"
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/VRMLnodes/SoVRMLExtrusion.h>
#include <Inventor/nodes/SoRotation.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/SbLinear.h>
#include <H5Cpp.h>
#include <hdf5serie/vectorserie.h>
#include <QtGui/QMenu>

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
    CoilSpring(OpenMBV::Object* obj, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent);

    /** info string in spine extrusion pop-up menu */
    virtual QString getInfo();

  protected:
    /** material and color */
    SoMaterial *mat;

    /** extrusion body */
    SoVRMLExtrusion *extrusion;

    /** translation of helix */
    SoTranslation* fromPoint;

    /** rotation of helix */
    SoRotation* rotation;

    /** memory for efficient spine update */ 
    float* spine;

    /** radius of helix */
    double springRadius;

    /** mumber of coils */
    int numberOfCoils;

    /** number of spine points */
    static const int numberOfSpinePoints = 120;

    /** cross section resolution */
    static const int iCircSegments = 20;

    /** scale factor */
    double scaleValue;

    /** local h5 data set copy */
    H5::VectorSerie<double> *h5Data;

    /** update method invoked at each time step */
    virtual double update();
};

#endif /* _COILSPRING_H_ */

