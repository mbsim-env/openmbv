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
#include <QMenu>
#include <cfloat>

namespace OpenMBVGUI {

Frame::Frame(const boost::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  f=boost::static_pointer_cast<OpenMBV::Frame>(obj);
  iconFile="frame.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // create so
  soSepRigidBody->addChild(Utils::soFrame(f->getSize(), f->getOffset(), true));
  // scale ref/localFrame
  refFrameScale->scaleFactor.setValue(f->getSize()*f->getScaleFactor(),f->getSize()*f->getScaleFactor(),f->getSize()*f->getScaleFactor());
  localFrameScale->scaleFactor.setValue(f->getSize()*f->getScaleFactor(),f->getSize()*f->getScaleFactor(),f->getSize()*f->getScaleFactor());
}

void Frame::createProperties() {
  RigidBody::createProperties();

  // GUI editors
  if(!clone) {
    properties->updateHeader();
    FloatEditor *sizeEditor=new FloatEditor(properties, QIcon(), "Size (length)");
    sizeEditor->setRange(0, DBL_MAX);
    sizeEditor->setOpenMBVParameter(f, &OpenMBV::Frame::getSize, &OpenMBV::Frame::setSize);

    FloatEditor *offsetEditor=new FloatEditor(properties, QIcon(), "Offset");
    offsetEditor->setRange(0, 1);
    offsetEditor->setStep(0.02);
    offsetEditor->setOpenMBVParameter(f, &OpenMBV::Frame::getOffset, &OpenMBV::Frame::setOffset);
  }
}

}
