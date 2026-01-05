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

#ifndef _OPENMBVGUI_SOCARDANROTATIONENGINE_H_
#define _OPENMBVGUI_SOCARDANROTATIONENGINE_H_

#include <Inventor/C/errors/debugerror.h> // workaround a include order bug in Coin-3.1.3
#include <Inventor/engines/SoSubEngine.h>
#include <Inventor/fields/SoSFVec3f.h>
#include <Inventor/fields/SoSFBool.h>

namespace OpenMBVGUI {

class CardanRotationEngine : public SoEngine {
  SO_ENGINE_HEADER(CardanRotationEngine);
  public:
    SoSFVec3f angle;
    SoSFBool inverse;
    SoEngineOutput rotation;
 
    static void initClass();
    CardanRotationEngine();
 
  private:
    ~CardanRotationEngine() override = default;
    void evaluate() override;
};

}

#endif
