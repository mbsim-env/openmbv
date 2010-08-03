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

#include "config.h"
#include "frame.h"
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include "utils.h"
#include "openmbvcppinterface/frame.h"

Frame::Frame(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent) : RigidBody(obj, parentItem, soParent) {
  OpenMBV::Frame *f=(OpenMBV::Frame*)obj;
  iconFile=":/frame.svg";
  setIcon(0, Utils::QIconCached(iconFile.c_str()));

  // create so
  soSepRigidBody->addChild(Utils::soFrame(f->getSize(), f->getOffset(), true));
  // scale ref/localFrame
  refFrameScale->scaleFactor.setValue(f->getSize(),f->getSize(),f->getSize());
  localFrameScale->scaleFactor.setValue(f->getSize(),f->getSize(),f->getSize());
}
