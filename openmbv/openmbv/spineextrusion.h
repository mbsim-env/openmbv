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

#ifndef _OPENMBVGUI_SPINEEXTRUSION_H_
#define _OPENMBVGUI_SPINEEXTRUSION_H_

#include "SoSpecial.h"
#include "dynamiccoloredbody.h"
#include <Inventor/C/errors/debugerror.h> // workaround a include order bug in Coin-3.1.3
#include <Inventor/VRMLnodes/SoVRMLExtrusion.h>
#include <Inventor/SbLinear.h>
#include <Inventor/fields/SoMFMatrix.h>
#include <Inventor/nodes/SoShaderParameter.h>
#include <hdf5serie/vectorserie.h>
#include <QMenu>

namespace OpenMBV {
  class SpineExtrusion;
}

class SoNormal;

namespace OpenMBVGUI {

class ExtrusionCardan {
  public:
    void init(int spSize, const std::shared_ptr<std::vector<std::shared_ptr<OpenMBV::PolygonPoint> > > &contour,
              double csScale, bool ccw,
              SoSeparator *soSep, SoSeparator *soOutLineSep);
    void setCardanWrtWorldSpine(const std::vector<OpenMBV::Float>& data, bool updateNormals=true);
  private:
    std::vector<SbVec3f> nsp;
    std::vector<SbVec3f> normal;
    SoCoordinate3 *quadMeshCoords;
    SoNormal *quadMeshNormals;
    SoTranslation *endCupTrans[2];
    SoRotation *endCupRot[2];
};

class ExtrusionCardanShader {
  public:
    void init(int NSp, SoMaterial *mat, double csScale, bool ccw,
              const std::shared_ptr<std::vector<std::shared_ptr<OpenMBV::PolygonPoint> > > &contour, SoSeparator *soSep);
    void updateData(const std::vector<OpenMBV::Float>& data);
    void pickUpdate(const std::vector<OpenMBV::Float>& data);
    void pickUpdateRestore();
  private:
    SoShaderParameterArray1f *dataNodeVector;
    int Nsp;
    double csScale;
    std::shared_ptr<std::vector<std::shared_ptr<OpenMBV::PolygonPoint> > > contour;
    SoCoordinate3 *vertex;
    SepNoPickNoBBox *sepNoPickNoBBox;
};

/**
 * \brief class for extrusion along a curve
 * \author Thorsten Schindler
 * \date 2009-05-06 initial commit (Thorsten Schindler)
 * \date 2014-01-29 initial twist (Thorsten Schindler)
 */
class SpineExtrusion : public DynamicColoredBody {
  Q_OBJECT
  public:
    /** constructor */
    SpineExtrusion(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind);
    ~SpineExtrusion();

    /** info string in spine extrusion pop-up menu */
    QString getInfo() override;

  protected:
    /** extrusion body */
    SoVRMLExtrusion *extrusion;

    /** number of spine points */
    int numberOfSpinePoints;

    /** twist axis */
    SbVec3f twistAxis;
  
    /** update method invoked at each time step */
    double update() override;
    void pickUpdate() override;
    void pickUpdateRestore() override;

    /** test for collinear spine points */
    bool collinear;

    /** additional twist because of collinear spine points */
    double additionalTwist;

    std::shared_ptr<OpenMBV::SpineExtrusion> spineExtrusion;
    void createProperties() override;

    int doublesPerPoint;

    void setIvSpine(const std::vector<OpenMBV::Float>& data);

    ExtrusionCardan extrusionCardan;
    ExtrusionCardanShader extrusionCardanShader;
    std::vector<OpenMBV::Float> data;
};

}

#endif /* _SPINEEXTRUSION_H_ */

