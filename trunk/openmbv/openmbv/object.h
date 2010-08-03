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

#ifndef _OBJECT_H_
#define _OBJECT_H_

#include "config.h"
#include <openmbvcppinterface/object.h>
#include <QtGui/QTreeWidgetItem>
#include <string>
#include <vector>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/fields/SoSFUInt32.h>
#include <H5Cpp.h>
#include <map>
#include <Inventor/sensors/SoNodeSensor.h>
#include <Inventor/nodes/SoTranslation.h>

namespace OpenMBV {
  class Object;
}

class Object : public QObject, public QTreeWidgetItem {
  Q_OBJECT
  protected:
    OpenMBV::Object *object;
    SoSwitch *soSwitch;
    SoSeparator *soSep;
    QAction *draw;
    bool drawThisPath;
    SoSwitch *soBBoxSwitch;
    SoSeparator *soBBoxSep;
    SoTranslation *soBBoxTrans;
    SoCube *soBBox;
    QAction *bbox;
    std::string iconFile;
    static std::map<SoNode*,Object*> objectMap;
  public:
    Object(OpenMBV::Object* obj, QTreeWidgetItem *parentItem, SoGroup *soParent);
    virtual QMenu* createMenu();
    void setEnableRecursive(bool enable);
    std::string getPath();
    std::string &getIconFile() { return iconFile; }
    virtual QString getInfo();
    static void nodeSensorCB(void *data, SoSensor*);
    static std::map<SoNode*,Object*>& getObjectMap() { return objectMap; }
  public slots:
    void drawSlot();
    void bboxSlot();
};

#endif
