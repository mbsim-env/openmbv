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

#include "dynamiccoloredbody.h"
#include <Inventor/C/errors/debugerror.h> // workaround a include order bug in Coin-3.1.3
#include <Inventor/VRMLnodes/SoVRMLExtrusion.h>
#include <Inventor/SbLinear.h>
#include <hdf5serie/vectorserie.h>
#include <QMenu>

namespace OpenMBV {
  class SpineExtrusion;
}

namespace OpenMBVGUI {

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

    /** test for collinear spine points */
    bool collinear;

    /** additional twist because of collinear spine points */
    double additionalTwist;

    std::shared_ptr<OpenMBV::SpineExtrusion> spineExtrusion;
    void createProperties() override;

    void setIvSpine(const std::vector<double>& data);
};

}

#endif /* _SPINEEXTRUSION_H_ */

