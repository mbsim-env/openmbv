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

#ifndef _OPENMBVGUI_INDEXEDTESSELATIONFACE_H_
#define _OPENMBVGUI_INDEXEDTESSELATIONFACE_H_

#include <Inventor/C/errors/debugerror.h> // workaround a include order bug in Coin-3.1.3
#include <Inventor/fields/SoMFInt32.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/fields/SoMFVec3d.h>
#include <Inventor/fields/SoSFEnum.h>

namespace OpenMBVGUI {

class IndexedTesselationFace : public SoSeparator {
 SO_NODE_HEADER(IndexedTesselationFace);
 public:
   SoSFEnum windingRule;
   SoMFVec3d coordinate;
   SoMFInt32 coordIndex;

   static void initClass();
   IndexedTesselationFace();
   IndexedTesselationFace(int numChilderen);
   enum WindingRule { ODD, NONZERO, POSITIVE, NEGATIVE, ABS_GEQ_TWO };

   void write(SoWriteAction *action) override;

   // This function must be called after all attributes are set.
   // When reading from a file it is automatically called.
   // This is a HACK because IndexedTesselationFace is not clearly implemented.
   void generate() { readChildren(nullptr); }

 protected:
   SbBool readChildren(SoInput *in) override;

 private:
   ~IndexedTesselationFace() override;
   void constructor();
};

}

#endif
