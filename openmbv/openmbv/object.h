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

#ifndef _OPENMBVGUI_OBJECT_H_
#define _OPENMBVGUI_OBJECT_H_

#include <fmatvec/atom.h>
#include <QtGui/QTreeWidgetItem>
#include <string>
#include <set>
#include <Inventor/C/errors/debugerror.h> // workaround a include order bug in Coin-3.1.3
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoGroup.h>
#include <Inventor/sensors/SoSensor.h>
#include <Inventor/sensors/SoNodeSensor.h>


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
    SoTranslation *soBBoxTrans;
    SoCube *soBBox;
    std::string iconFile;
    SoNodeSensor *nodeSensor;
    PropertyDialog *properties;
    Object *clone;
    Object *getClone();
    static std::set<Object*> objects;
    bool getBoundingBox();
    BoolEditor *boundingBoxEditor;
    virtual void createProperties();
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
};

}

#endif
