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

#ifndef _OPENMBVGUI_OBJECT_H_
#define _OPENMBVGUI_OBJECT_H_

#include "openmbvcppinterface/object.h"
#include <fmatvec/atom.h>
#include <QTreeWidgetItem>
#include <string>
#include <set>
#include <Inventor/nodes/SoDrawStyle.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoMatrixTransform.h>
#include <Inventor/nodes/SoGroup.h>
#include <Inventor/sensors/SoSensor.h>
#include <Inventor/sensors/SoNodeSensor.h>
#include <Inventor/nodes/SoBaseColor.h>


namespace OpenMBV {
  class Object;
}

namespace OpenMBVGUI {

class PropertyDialog;
class BoolEditor;

class Object : public QObject, public QTreeWidgetItem, virtual public fmatvec::Atom {
  Q_OBJECT
  friend class Editor;
  friend class MainWindow;
  friend class RigidBody;
  protected:
    std::shared_ptr<OpenMBV::Object> object;
    SoSwitch *soSwitch;
    SoSeparator *soSep;
    bool drawThisPath;
    SoSwitch *soBBoxSwitch;
    SoSeparator *soBBoxSep;
    SoMatrixTransform *soBBoxTrans;
    SoCube *soBBox;
    std::string iconFile;
    SoNodeSensor *nodeSensor;
    PropertyDialog *properties;
    Object *clone;
    static std::set<Object*> objects;
    BoolEditor *boundingBoxEditor;
    virtual void createProperties();
    bool highlight { false };
    void setHighlight(bool value);
    bool drawBoundingBox() { return highlight || object->getBoundingBox(); }
  public:
    Object(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind);
    ~Object() override;
    std::string &getIconFile() { return iconFile; }
    std::shared_ptr<OpenMBV::Object> getObject() { return object; }
    virtual QString getInfo();
    static void nodeSensorCB(void *data, SoSensor*);
    PropertyDialog *getProperties();
    void deleteObjectSlot();
    void setBoundingBox(bool value);
  private:
    void replaceBBoxHighlight();
    bool isCloneToBeDeleted { false };
};

}

#endif
