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

#ifndef _OPENMBVGUI_GROUP_H_
#define _OPENMBVGUI_GROUP_H_

#include "object.h"
#include <string>

// If Coin and SoQt is linked as a dll no symbols of this file are exported (for an unknown reason).
// Hence we explicitly export ALL symbols.
// We cannot export selectively symbols since some symbols defined by Q_OBJECT needs also to be exported.
// The clear Qt way would be to use PImpl but this is not done here.
#ifdef _WIN32
#  define DLL_PUBLIC __declspec(dllexport)
#else
#  define DLL_PUBLIC
#endif

namespace OpenMBV {
  class Group;
}

namespace OpenMBVGUI {

class DLL_PUBLIC Group : public Object {
  Q_OBJECT
  friend class MainWindow;
  friend class Object;
  protected:
    virtual void update() {}
    std::shared_ptr<OpenMBV::Group> grp;
    void createProperties() override;
  public:
    Group(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind);
    QString getInfo() override;
    void newObjectSlot();
    void reloadFileSlot();
    void unloadFileSlot();

    /** Request a flush of the writer.
     * If a writer process currently exists true is returned else false. Note that this flag cannot change
     * while the calling process as opened the file for reading.
     */
    bool requestFlush();
  Q_SIGNALS:
    // just a signal to call reloadFileSlot from an arbitary thread.
    void reloadFileSignal();
    void refreshFileSignal();
};

}

#endif
