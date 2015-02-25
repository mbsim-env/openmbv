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

#ifndef _OPENMBVGUI_GROUP_H_
#define _OPENMBVGUI_GROUP_H_

#include "object.h"
#include <string>
#include <boost/date_time/posix_time/posix_time.hpp>

// If Coin and SoQt is linked as a dll no symbols of this file are exported (for an unknown reason).
// Hence we explicitly export the required symbols. This should be done for all code for a clean Windows build!
#ifdef _WIN32
#  define DLL_PUBLIC __declspec(dllexport)
#else
#  define DLL_PUBLIC
#endif

namespace OpenMBV {
  class Group;
}

namespace OpenMBVGUI {

class Group : public Object {
  Q_OBJECT
  friend class MainWindow;
  friend class Object;
  protected:
    virtual void update() {}
    boost::shared_ptr<OpenMBV::Group> grp;
    QTimer *reloadTimer;
    boost::posix_time::ptime xmlLastModified, h5LastModified;
    void createProperties();
  public:
    Group(const boost::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind);
    virtual QString getInfo();
  protected slots:
    void newObjectSlot();
    void saveFileSlot();
    void reloadFileSlotIfNewer();
    void reloadFileSlot();
  public slots:
    DLL_PUBLIC void unloadFileSlot();
};

}

#endif
