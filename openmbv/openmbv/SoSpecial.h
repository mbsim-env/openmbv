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

#ifndef _OPENMBVGUI_SOSEPNOPICKNOBBOX_H_
#define _OPENMBVGUI_SOSEPNOPICKNOBBOX_H_

#include <Inventor/C/errors/debugerror.h> // workaround a include order bug in Coin-3.1.3
#include <Inventor/fields/SoSFBool.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/elements/SoOverrideElement.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/actions/SoGLRenderAction.h>

namespace OpenMBVGUI {

class SepNoPickNoBBox : public SoSeparator {
  SO_NODE_HEADER(SepNoPickNoBBox);
  public:
    SepNoPickNoBBox();
    static void initClass();
    void rayPick(SoRayPickAction *action) override;
    void getBoundingBox(SoGetBoundingBoxAction *action) override;
    SoSFBool skipBBox;
    SoSFBool skipPick;
};

class SepNoPick : public SoSeparator {
  SO_NODE_HEADER(SepNoPick);
  public:
    SepNoPick();
    static void initClass();
    void rayPick(SoRayPickAction *action) override {}
};

// equals SoBaseColor but the color is uses even if the override flag is set
class BaseColorHeavyOverride : public SoBaseColor {
  SO_NODE_HEADER(BaseColorHeavyOverride);
  public:
    BaseColorHeavyOverride();
    static void initClass();
    void GLRender(SoGLRenderAction *action) override {
      SoState *state=action->getState(); // get state
      SoOverrideElement::setDiffuseColorOverride(state, this, false); // disable override
      SoBaseColor::GLRender(action); // render
    }
};

}

#endif
