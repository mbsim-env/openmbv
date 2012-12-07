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

#ifndef _MBSIMGUI_OBJECT_H_
#define _MBSIMGUI_OBJECT_H_

#include <QtGui/QTreeWidgetItem>
#include <string>
#include <set>
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

class Object : public QObject, public QTreeWidgetItem {
  Q_OBJECT
  friend class Editor;
  friend class MainWindow;
  protected:
    OpenMBV::Object *object;
    SoSwitch *soSwitch;
    SoSeparator *soSep;
    bool drawThisPath;
    SoSwitch *soBBoxSwitch;
    SoSeparator *soBBoxSep;
    SoTranslation *soBBoxTrans;
    SoCube *soBBox;
    std::string iconFile;
    bool searchMatched;
    SoNodeSensor *nodeSensor;
    PropertyDialog *properties;
    Object *clone;
    Object *getClone();
    static std::set<Object*> objects;
    QMenu *contextMenu;
  public:
    Object(OpenMBV::Object* obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind);
    virtual ~Object();
    virtual std::string getPath();
    std::string &getIconFile() { return iconFile; }
    virtual QString getInfo();
    static void nodeSensorCB(void *data, SoSensor*);
    void updateTextColor();
    bool getSearchMatched() { return searchMatched; }
    void setSearchMatched(bool m) { searchMatched=m; }
  protected slots:
    void deleteObjectSlot();
};

}

#endif
