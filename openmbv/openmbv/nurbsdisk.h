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

#ifndef _OPENMBVGUI_NURBSDISK_H_
#define _OPENMBVGUI_NURBSDISK_H_

#include "dynamiccoloredbody.h"

#include <Inventor/C/errors/debugerror.h> // workaround a include order bug in Coin-3.1.3
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoRotation.h>
#include <Inventor/nodes/SoRotationXYZ.h>
#include <Inventor/SoPrimitiveVertex.h>
#include <Inventor/nodes/SoIndexedNurbsSurface.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoShapeHints.h>
#include <Inventor/nodes/SoSeparator.h>

#include <hdf5serie/vectorserie.h>
#include <QMenu>

namespace OpenMBV {
  class NurbsDisk;
}

namespace OpenMBVGUI {

/**
 * \brief class for bodies with NURBS surface and primitive closure
 * \author Kilian Grundl
 * \author Raphael Missel
 * \author Thorsten Schindler
 * \date 2009-05-20 initial commit (Grundl / Missel / Schindler)
 * \date 2009-08-17 visualisation / contour (Grundl / Missel / Schindler)
 * \date 2010-08-09 adapt to new concept of Markus Friedrich (Schindler)
 * \todo face sets TODO
 */
class NurbsDisk : public DynamicColoredBody {
  Q_OBJECT
  public:
    /** constructor */
    NurbsDisk(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind);

    ~NurbsDisk() override;

    /** info string in spine extrusion pop-up menu */
    QString getInfo() override;

  protected:
    SoSwitch *soLocalFrameSwitch;
    SoScale *localFrameScale;
    SoSeparator *soSepNurbsDisk;

    SoRotationXYZ *rotationAlpha, *rotationBeta, *rotationGamma;
    SoRotation *rotation; // accumulated rotationAlpha, rotationBeta and rotationGamma
    SoTranslation *translation;

    /** number of elements in azimuthal and radial direction */
    int nj, nr;

    /** interpolation degree radial and azimuthal */
    int degRadial, degAzimuthal;

    /** number of intermediate points between finite element nodes for visualisation */
    int drawDegree;

    /** knot vector azimuthal and radial */
    std::vector<double> knotVecAzimuthal, knotVecRadial;

    /** inner and outer radius */
    float innerRadius, outerRadius;

    /** number of nurbs control points */
    int nurbsLength;

    /** NURBS surface */
    SoIndexedNurbsSurface *surface;

    /**
     * \brief NURBS control-points and points for sides and back (=midplane) of the disk
     *
     * For the indexing of these points:
     * The first points (with the number of "nurbsLength") in this array are the control points of the NURBS-surface
     * Besides the points for the side/flank/facet are saved:
     *    - at first the points of the inner ring with its points on the surface and the points on the midplane
     *    - then the points of the outer ring with its points on the surface and the points on the midplane
     */
    SoCoordinate3 *controlPts;

    /** primitive closures */
    SoIndexedFaceSet *faceSet;

    /** update method invoked at each time step */
    double update() override;

    std::shared_ptr<OpenMBV::NurbsDisk> nurbsDisk;
    void createProperties() override;

    public:
      void moveCameraWithSlot();
};

}

#endif /* _NURBSDISK_H_ */

