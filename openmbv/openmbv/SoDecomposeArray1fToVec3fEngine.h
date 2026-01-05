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

#ifndef _OPENMBVGUI_SODECOMPOSEARRAY1FTOVEC3F_H_
#define _OPENMBVGUI_SODECOMPOSEARRAY1FTOVEC3F_H_

#include <Inventor/C/errors/debugerror.h> // workaround a include order bug in Coin-3.1.3
#include <Inventor/engines/SoSubEngine.h>
#include <Inventor/fields/SoSFInt32.h>
#include <Inventor/fields/SoMFFloat.h>

namespace OpenMBVGUI {

class DecomposeArray1fToVec3fEngine : public SoEngine {
  SO_ENGINE_HEADER(DecomposeArray1fToVec3fEngine);
  public:
    SoSFInt32 startIndex;
    SoMFFloat input;
    SoEngineOutput output;
 
    static void initClass();
    DecomposeArray1fToVec3fEngine();
 
  private:
    ~DecomposeArray1fToVec3fEngine() override = default;
    void evaluate() override;
};

}

#endif
