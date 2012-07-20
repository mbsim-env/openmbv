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

#ifndef _SOSEPNOPICKNOBBOX_H_
#define _SOSEPNOPICKNOBBOX_H_

#include "config.h"
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/elements/SoOverrideElement.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/actions/SoGLRenderAction.h>

class SoSepNoPickNoBBox : public SoSeparator {
  public:
    void rayPick(SoRayPickAction *action) {}
    void getBoundingBox(SoGetBoundingBoxAction *action) {}
};

// equals SoBaseColor but the color is uses even if the override flag is set
class SoBaseColorHeavyOverride : public SoBaseColor {
  public:
    void GLRender(SoGLRenderAction *action) {
      SoState *state=action->getState(); // get state
      SoOverrideElement::setDiffuseColorOverride(state, this, false); // disable override
      SoBaseColor::GLRender(action); // render
    }
};

#endif
