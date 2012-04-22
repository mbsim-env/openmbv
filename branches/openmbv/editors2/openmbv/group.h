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

#ifndef _GROUP_H_
#define _GROUP_H_

#include "config.h"
#include "object.h"
#include <string>
#include <H5Cpp.h>
#include <QtCore/QFileInfo>
#include <QtCore/QDateTime>

namespace OpenMBV {
  class Group;
}

class Group : public Object {
  Q_OBJECT
  friend class MainWindow;
  friend class Object;
  protected:
    virtual void update() {}
    QAction *saveFile, *unloadFile, *reloadFile;
    OpenMBV::Group *grp;
    QTimer *reloadTimer;
    QFileInfo *xmlFileInfo, *h5FileInfo;
    QDateTime xmlLastModified, h5LastModified;
    std::string getPath();
  public:
    Group(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind);
    virtual QString getInfo();
  protected slots:
    void saveFileSlot();
    void reloadFileSlotIfNewer();
    void reloadFileSlot();
  public slots:
    void unloadFileSlot();
};

#endif
