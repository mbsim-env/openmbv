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
#include "group.h"
#include <QtGui/QMenu>
#include <QtGui/QPushButton>
#include "objectfactory.h"
#include "mainwindow.h"
#include <string>
#include "utils.h"
#include <QtGui/QMessageBox>
#include <boost/functional/factory.hpp>
#include "openmbvcppinterface/arrow.h"
#include "openmbvcppinterface/coilspring.h"
#include "openmbvcppinterface/nurbsdisk.h"
#include "openmbvcppinterface/compoundrigidbody.h"
#include "openmbvcppinterface/cube.h"
#include "openmbvcppinterface/cuboid.h"
#include "openmbvcppinterface/extrusion.h"
#include "openmbvcppinterface/frame.h"
#include "openmbvcppinterface/frustum.h"
#include "openmbvcppinterface/grid.h"
#include "openmbvcppinterface/invisiblebody.h"
#include "openmbvcppinterface/ivbody.h"
#include "openmbvcppinterface/rotation.h"
#include "openmbvcppinterface/sphere.h"
#include "openmbvcppinterface/spineextrusion.h"
#include "openmbvcppinterface/path.h"
#include "openmbvcppinterface/group.h"

/* This is a varaint of the boost::filesystem::last_write_time functions.
 * It only differs in the argument/return value being here a boost::posix_time::ptime instead of a time_t.
 * This enables file timestamps on microsecond level. */
#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#endif
namespace boost {
  namespace myfilesystem {
    boost::posix_time::ptime last_write_time(const boost::filesystem::path &p) {
#ifndef _WIN32
      struct stat st;
      if(stat(p.generic_string().c_str(), &st)!=0)
        throw boost::filesystem::filesystem_error("system stat call failed", p, boost::system::error_code());
      boost::posix_time::ptime time;
      time=boost::posix_time::from_time_t(st.st_mtime);
      time+=boost::posix_time::microsec(st.st_mtim.tv_nsec/1000);
      return time;
#else
      HANDLE f=CreateFile(p.generic_string().c_str(), GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
      if(f==INVALID_HANDLE_VALUE)
        throw boost::filesystem::filesystem_error("CreateFile failed", p, boost::system::error_code());
      FILETIME create, lastAccess, lastWrite;
      if(GetFileTime(f, &create, &lastAccess, &lastWrite)==0) {
        CloseHandle(f);
        throw boost::filesystem::filesystem_error("GetFileTime failed", p, boost::system::error_code());
      }
      CloseHandle(f);
      uint64_t microSecSince1601=((((uint64_t)(lastWrite.dwHighDateTime))<<32)+lastWrite.dwLowDateTime)/10;
      uint64_t hoursSince1601=microSecSince1601/1000000/60/60;
      return boost::posix_time::ptime(boost::gregorian::date(1601,boost::gregorian::Jan,1),
                                      boost::posix_time::hours(hoursSince1601)+
                                      boost::posix_time::microseconds(microSecSince1601-hoursSince1601*60*60*1000000));
#endif
    }
    void last_write_time(const boost::filesystem::path &p, const boost::posix_time::ptime &time) {
#ifndef _WIN32
      struct timeval times[2];
      boost::posix_time::time_period sinceEpoch(boost::posix_time::ptime(boost::gregorian::date(1970, boost::gregorian::Jan, 1)), time);
      times[0].tv_sec =sinceEpoch.length().total_seconds();
      times[0].tv_usec=sinceEpoch.length().total_microseconds()-1000000*times[0].tv_sec;
      times[1].tv_sec =times[0].tv_sec;
      times[1].tv_usec=times[0].tv_usec;
      if(utimes(p.generic_string().c_str(), times)!=0)
        throw boost::filesystem::filesystem_error("system utimes call failed", p, boost::system::error_code());
#else
      HANDLE f=CreateFile(p.generic_string().c_str(), FILE_WRITE_ATTRIBUTES, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
      if(f==INVALID_HANDLE_VALUE)
        throw boost::filesystem::filesystem_error("CreateFile failed", p, boost::system::error_code());
      boost::posix_time::time_period since1601(boost::posix_time::ptime(boost::gregorian::date(1601, boost::gregorian::Jan, 1)), time);
      boost::posix_time::time_duration dt=since1601.length();
      uint64_t winTime=((uint64_t)(dt.hours()))*60*60*10000000;
      dt-=boost::posix_time::hours(dt.hours());
      winTime+=dt.total_microseconds()*10;
      FILETIME changeTime;
      changeTime.dwHighDateTime=(winTime>>32);
      changeTime.dwLowDateTime=(winTime & ((((uint64_t)1)<<32)-1));
      if(SetFileTime(f, NULL, &changeTime, &changeTime)==0) {
        CloseHandle(f);
        throw boost::filesystem::filesystem_error("SetFileTime failed", p, boost::system::error_code());
      }
      CloseHandle(f);
#endif
    }
  }
}

using namespace std;

namespace OpenMBVGUI {

Group::Group(OpenMBV::Object* obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : Object(obj, parentItem, soParent, ind) {
  grp=(OpenMBV::Group*)obj;
  iconFile="group.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // expand or collapse
  setExpanded(grp->getExpand());

  // if xml:base attribute exist => new sub file
  if(grp->getSeparateFile()) {
    iconFile="h5file.svg";
    setIcon(0, Utils::QIconCached(iconFile));
    setText(0, grp->getFileName().c_str());
    setFlags(flags() & ~Qt::ItemIsEditable);
  }
  // read XML
  vector<OpenMBV::Object*> child=grp->getObjects();
  for(unsigned int i=0; i<child.size(); i++) {
    if(child[i]->getClassName()=="Group" && ((OpenMBV::Group*)child[i])->getObjects().size()==0) continue; // a hack for openmbvdeleterows.sh
    ObjectFactory(child[i], this, soSep, -1);
  }

  // GUI
  QAction *newObject=new QAction(Utils::QIconCached("newobject.svg"),"Create new Object", this);
  connect(newObject,SIGNAL(triggered()),this,SLOT(newObjectSlot()));
  properties->addContextAction(newObject);

  if(grp->getSeparateFile()) {
    saveFile=new QAction(Utils::QIconCached("savefile.svg"),"Save XML-file", this);
    saveFile->setObjectName("Group::saveFile");
    connect(saveFile,SIGNAL(triggered()),this,SLOT(saveFileSlot()));
    properties->addContextAction(saveFile);

    unloadFile=new QAction(Utils::QIconCached("unloadfile.svg"),"Unload XML/H5-file", this);
    unloadFile->setObjectName("Group::unloadFile");
    connect(unloadFile,SIGNAL(triggered()),this,SLOT(unloadFileSlot()));
    properties->addContextAction(unloadFile);

    reloadFile=new QAction(Utils::QIconCached("reloadfile.svg"),"Reload XML/H5-file", this);
    reloadFile->setObjectName("Group::reloadFile");
    connect(reloadFile,SIGNAL(triggered()),this,SLOT(reloadFileSlot()));
    properties->addContextAction(reloadFile);
  }

  // timer for reloading file automatically
  reloadTimer=NULL;
  // if reloading is enabled and this Group is a toplevel file create timer
  if(grp->getParent()==NULL && MainWindow::getInstance()->getReloadTimeout()>0) {
    xmlLastModified=boost::myfilesystem::last_write_time(text(0).toStdString().c_str());
    h5LastModified =boost::myfilesystem::last_write_time((text(0).remove(text(0).count()-3, 3)+"h5").toStdString().c_str());

    reloadTimer=new QTimer(this);
    connect(reloadTimer,SIGNAL(timeout()),this,SLOT(reloadFileSlotIfNewer()));
    reloadTimer->start(MainWindow::getInstance()->getReloadTimeout());
  }

  if(!clone) {
    properties->updateHeader();
    // GUI editors (none)
  }
}

QString Group::getInfo() {
  return Object::getInfo()+
         QString("<hr width=\"10000\"/>")+
         QString("<b>Number of children:</b> %1").arg(childCount());
}

void Group::newObjectSlot() {
  static vector<Utils::FactoryElement> factory=boost::assign::list_of
    (Utils::FactoryElement(Utils::QIconCached("arrow.svg"),             "Arrow",             boost::factory<OpenMBV::Arrow*>()))
    (Utils::FactoryElement(Utils::QIconCached("coilspring.svg"),        "CoilSpring",        boost::factory<OpenMBV::CoilSpring*>()))
    (Utils::FactoryElement(Utils::QIconCached("invisiblebody.svg"),     "NurbsDisk",         boost::factory<OpenMBV::NurbsDisk*>()))
    (Utils::FactoryElement(Utils::QIconCached("compoundrigidbody.svg"), "CompoundRigidBody", boost::factory<OpenMBV::CompoundRigidBody*>()))
    (Utils::FactoryElement(Utils::QIconCached("cube.svg"),              "Cube",              boost::factory<OpenMBV::Cube*>()))
    (Utils::FactoryElement(Utils::QIconCached("cuboid.svg"),            "Cuboid",            boost::factory<OpenMBV::Cuboid*>()))
    (Utils::FactoryElement(Utils::QIconCached("extrusion.svg"),         "Extrusion",         boost::factory<OpenMBV::Extrusion*>()))
    (Utils::FactoryElement(Utils::QIconCached("frame.svg"),             "Frame",             boost::factory<OpenMBV::Frame*>()))
    (Utils::FactoryElement(Utils::QIconCached("frustum.svg"),           "Frustum",           boost::factory<OpenMBV::Frustum*>()))
    (Utils::FactoryElement(Utils::QIconCached("invisiblebody.svg"),     "Grid",              boost::factory<OpenMBV::Grid*>()))
    (Utils::FactoryElement(Utils::QIconCached("invisiblebody.svg"),     "InvisibleBody",     boost::factory<OpenMBV::InvisibleBody*>()))
    (Utils::FactoryElement(Utils::QIconCached("ivbody.svg"),            "IvBody",            boost::factory<OpenMBV::IvBody*>()))
    (Utils::FactoryElement(Utils::QIconCached("rotation.svg"),          "Rotation",          boost::factory<OpenMBV::Rotation*>()))
    (Utils::FactoryElement(Utils::QIconCached("sphere.svg"),            "Sphere",            boost::factory<OpenMBV::Sphere*>()))
    (Utils::FactoryElement(Utils::QIconCached("invisiblebody.svg"),     "SpineExtrusion",    boost::factory<OpenMBV::SpineExtrusion*>()))
    (Utils::FactoryElement(Utils::QIconCached("path.svg"),              "Path",              boost::factory<OpenMBV::Path*>()))
    (Utils::FactoryElement(Utils::QIconCached("group.svg"),             "Group",             boost::factory<OpenMBV::Group*>()))
  .to_container(factory);  

  vector<string> existingNames;
  for(unsigned int j=0; j<grp->getObjects().size(); j++)
    existingNames.push_back(grp->getObjects()[j]->getName());

  OpenMBV::Object *obj=Utils::createObjectEditor(factory, existingNames, "Create new Object");
  if(obj==NULL) return;

  grp->addObject(obj);
  ObjectFactory(obj, this, soSep, -1);

  // apply object filter
  MainWindow::getInstance()->searchObjectList(this, QRegExp(MainWindow::getInstance()->filter->text()));
}

void Group::saveFileSlot() {
  static QMessageBox *askSave=NULL;
  static QCheckBox *showAgain=NULL;
  if(!askSave) {
    askSave=new QMessageBox(QMessageBox::Question, "Save XML-File", QString(
        "Save current properties in XML-File.\n"
        "\n"
        "This will overwrite the following files:\n"
        "- OpenMBV-XML-file '%1'\n"
        "- OpenMBV-Parameter-XML-file '%2' (if exists)\n"
        "- all included OpenMBV-XML-Files\n"
        "- all dedicated OpenMBV-Parameter-XML-Files"
      ).arg(grp->getFileName().c_str()).arg((grp->getFileName().substr(0,grp->getFileName().length()-4)+".param.xml").c_str()),
      QMessageBox::Cancel | QMessageBox::SaveAll);
    showAgain=new QCheckBox("Do not show this dialog again");
    QGridLayout *layout=static_cast<QGridLayout*>(askSave->layout());
    layout->addWidget(showAgain, layout->rowCount(), 0, 1, layout->columnCount());
  }
  QMessageBox::StandardButton ret=QMessageBox::SaveAll;
  if(showAgain->checkState()==Qt::Unchecked)
    ret=static_cast<QMessageBox::StandardButton>(askSave->exec());
  if(ret==QMessageBox::SaveAll)
    grp->write(true, false);
}

void Group::unloadFileSlot() {
  OpenMBV::Group *grpPtr=grp;
  // deleting an QTreeWidgetItem will remove the item from the tree (this is safe at any time)
  delete this; // from now no element should be accessed thats why we have saveed the grp member
  // if grp has a parent, remove grp from parent and delete grp
  grpPtr->destroy(); // this does not use any member of Group, so we can call it after "detete this". We delete the OpenMBVCppInterface after the Object such that in the Object dtor the getPath is available
}

void Group::reloadFileSlot() {
  // save file name and ind of this in parent
  string fileName=text(0).toStdString();
  QTreeWidgetItem *parent=QTreeWidgetItem::parent();
  int ind=parent?
            parent->indexOfChild(this):
            MainWindow::getInstance()->getRootItemIndexOfChild(this);
  // unload
  unloadFileSlot(); // NOTE: this calls "delete this" !!!
  // load
  MainWindow::getInstance()->openFile(fileName, parent, parent?((Group*)parent)->soSep:NULL, ind);
  // set new item the current item: this selects and scroll to the new widget
  if(parent)
    MainWindow::getInstance()->objectList->setCurrentItem(parent->child(ind), 0, QItemSelectionModel::NoUpdate);
  else
    MainWindow::getInstance()->objectList->setCurrentItem(MainWindow::getInstance()->objectList->invisibleRootItem()->child(ind), 0, QItemSelectionModel::NoUpdate);

  emit MainWindow::getInstance()->fileReloaded();
}

void Group::reloadFileSlotIfNewer() {
  if(boost::myfilesystem::last_write_time(text(0).toStdString().c_str())>xmlLastModified &&
     boost::myfilesystem::last_write_time((text(0).remove(text(0).count()-3, 3)+"h5").toStdString().c_str())>h5LastModified) {
    xmlLastModified=boost::myfilesystem::last_write_time(text(0).toStdString().c_str());
    h5LastModified =boost::myfilesystem::last_write_time((text(0).remove(text(0).count()-3, 3)+"h5").toStdString().c_str());
    reloadFileSlot();
  }
}

string Group::getPath() {
  if(grp->getSeparateFile())
    return text(0).toStdString();
  return Object::getPath();
}

}
