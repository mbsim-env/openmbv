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
#include <QtGui/QTreeWidgetItem>
#include <string>
#include <vector>
#include "tinyxml.h"
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/fields/SoSFUInt32.h>
#include <H5Cpp.h>
#include <map>
#include <Inventor/sensors/SoNodeSensor.h>
#include <Inventor/nodes/SoTranslation.h>

#define OPENMBVNS "{http://openmbv.berlios.de/OpenMBV}"

// Use QIconCached(filename) instead of QIcon(filename) everywhere
// to cache the parsing of e.g. SVG files. This lead to a speedup
// (at app init) by a factor of 11 in my test case.
inline const QIcon& QIconCached(const QString& filename) {
  static std::map<QString, QIcon> myIconCache;
  std::map<QString, QIcon>::iterator i=myIconCache.find(filename);
  if(i==myIconCache.end())
    return myIconCache[filename]=QIcon(filename);
  return i->second;
}

inline SoGroup* SoDBreadAllCached(const std::string &filename) {
  static std::map<std::string, SoGroup*> myIvBodyCache;
  std::map<std::string, SoGroup*>::iterator i=myIvBodyCache.find(filename);
  if(i==myIvBodyCache.end()) {
    SoInput in;
    in.openFile(filename.c_str());
    return myIvBodyCache[filename]=SoDB::readAll(&in);
  }
  return i->second;
}

class Object : public QObject, public QTreeWidgetItem {
  Q_OBJECT
  protected:
    SoSwitch *soSwitch;
    SoSeparator *soSep;
    QAction *draw;
    bool drawThisPath;
    H5::Group *h5Group;
    SoSwitch *soBBoxSwitch;
    SoSeparator *soBBoxSep;
    SoTranslation *soBBoxTrans;
    SoCube *soBBox;
    QAction *bbox;
    std::string iconFile;
    static std::map<SoNode*,Object*> objectMap;
  public:
    Object(TiXmlElement* element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent);
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
