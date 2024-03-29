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

#ifndef _OPENMBVGUI_COMPOUNDRIGIDBODY_H_
#define _OPENMBVGUI_COMPOUNDRIGIDBODY_H_

#include "rigidbody.h"
#include <openmbvcppinterface/compoundrigidbody.h>

namespace OpenMBVGUI {

class CompoundRigidBody : public RigidBody {
  friend class MainWindow;
  Q_OBJECT
  public:
    CompoundRigidBody(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind);
    QString getInfo() override;
    void newRigidBodySlot();
  protected:
    double update() override;
    void createProperties() override;
  private:
    std::shared_ptr<OpenMBV::CompoundRigidBody> crb;
};

}

#endif
