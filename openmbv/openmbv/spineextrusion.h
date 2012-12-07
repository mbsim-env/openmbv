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

#ifndef _MBSIMGUI_SPINEEXTRUSION_H_
#define _MBSIMGUI_SPINEEXTRUSION_H_

#include "dynamiccoloredbody.h"
#include <Inventor/VRMLnodes/SoVRMLExtrusion.h>
#include <Inventor/SbLinear.h>
#include <H5Cpp.h>
#include <hdf5serie/vectorserie.h>
#include <QtGui/QMenu>

namespace OpenMBV {
  class SpineExtrusion;
}

namespace OpenMBVGUI {

/**
 * \brief class for extrusion along a curve
 * \author Thorsten Schindler
 * \date 2009-05-06 initial commit (Thorsten Schindler)
 */
class SpineExtrusion : public DynamicColoredBody {
  Q_OBJECT
  public:
    /** constructor */
    SpineExtrusion(OpenMBV::Object* obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind);

    /** info string in spine extrusion pop-up menu */
    virtual QString getInfo();

  protected:
    /** extrusion body */
    SoVRMLExtrusion *extrusion;

    /** number of spine points */
    int numberOfSpinePoints;

    /** twist axis */
    SbVec3f *twistAxis;
  
    /** update method invoked at each time step */
    virtual double update();

    OpenMBV::SpineExtrusion *spineExtrusion;
};

}

#endif /* _SPINEEXTRUSION_H_ */

