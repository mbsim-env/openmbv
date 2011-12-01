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
#include "objectfactory.h"
#include "mainwindow.h"
#include "compoundrigidbody.h"
#include "openmbvcppinterface/group.h"
#include <string>
#include "utils.h"
#include <QtGui/QMessageBox>

using namespace std;

Group::Group(OpenMBV::Object* obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : Object(obj, parentItem, soParent, ind) {
  grp=(OpenMBV::Group*)obj;
  iconFile=":/group.svg";
  setIcon(0, Utils::QIconCached(iconFile.c_str()));

  if(dynamic_cast<CompoundRigidBody*>(parentItem)==0) {
    // expand or collapse
    setExpanded(grp->getExpand());
  }

  // if xml:base attribute exist => new sub file
  if(grp->getSeparateFile()) {
    iconFile=":/h5file.svg";
    setIcon(0, Utils::QIconCached(iconFile.c_str()));
    setText(0, grp->getFileName().c_str());
  }
  // read XML
  vector<OpenMBV::Object*> child=grp->getObjects();
  for(unsigned int i=0; i<child.size(); i++) {
    if(child[i]->getClassName()=="Group" && ((OpenMBV::Group*)child[i])->getObjects().size()==0) continue; // a hack for openmbvdeleterows.sh
    ObjectFactory(child[i], this, soSep, -1);
  }

  // hide groups without childs
  if(childCount()==0) setHidden(true);

  // GUI
  saveFile=new QAction(Utils::QIconCached(":/savefile.svg"),"Save XML-File", this);
  saveFile->setObjectName("Group::saveFile");
  connect(saveFile,SIGNAL(activated()),this,SLOT(saveFileSlot()));

  if(grp->getParent()==NULL) {
    unloadFile=new QAction(Utils::QIconCached(":/unloadfile.svg"),"Unload XML/H5-File", this);
    unloadFile->setObjectName("Group::unloadFile");
    connect(unloadFile,SIGNAL(activated()),this,SLOT(unloadFileSlot()));

    reloadFile=new QAction(Utils::QIconCached(":/reloadfile.svg"),"Reload XML/H5-File", this);
    reloadFile->setObjectName("Group::reloadFile");
    connect(reloadFile,SIGNAL(activated()),this,SLOT(reloadFileSlot()));

    // timer for reloading file automatically
    reloadTimer=NULL;
    // if reloading is enabled and this Group is a separate file create timer
    if(MainWindow::getInstance()->getReloadTimeout()>0) {
      xmlFileInfo=new QFileInfo(text(0));
      h5FileInfo=new QFileInfo(text(0).remove(text(0).count()-3, 3)+"h5");
      xmlLastModified=xmlFileInfo->lastModified();
      h5LastModified=h5FileInfo->lastModified();
      reloadTimer=new QTimer(this);
      connect(reloadTimer,SIGNAL(timeout()),this,SLOT(reloadFileSlotIfNewer()));
      reloadTimer->start(MainWindow::getInstance()->getReloadTimeout());
    }
  }
}

QString Group::getInfo() {
  return Object::getInfo()+
         QString("<hr width=\"10000\"/>")+
         QString("<b>Number of Children:</b> %1").arg(childCount());
}

QMenu* Group::createMenu() {
  QMenu* menu=Object::createMenu();
  if(grp->getSeparateFile()) {
    menu->addSeparator()->setText("Properties from: Group");
    menu->addAction(saveFile);
  }
  if(grp->getParent()==NULL) {
    menu->addAction(unloadFile);
    menu->addAction(reloadFile);
  }
  return menu;
}

void Group::saveFileSlot() {
  if(QMessageBox::warning(0, "Overwrite XML-File", QString(
       "Save current object porperties in XML-File.\n"
       "\n"
       "Saving will overwrite the following files:\n"
       "- OpenMBV-XML-File '%1'\n"
       "- OpenMBV-Parameter-XML-file '%2'\n"
       "- all included OpenMBV-XML-Files\n"
       "- all dedicated OpenMBV-Parameter-XML-Files")
        .arg(grp->getFileName().c_str())
        .arg((grp->getFileName().substr(0,grp->getFileName().length()-4)+".param.xml").c_str()),
       QMessageBox::Save | QMessageBox::Cancel)==QMessageBox::Save)
    grp->write(true, false);
}

void Group::unloadFileSlot() {
  // if grp has a parent, remove grp from parent and delete grp
  grp->destroy();
  // deleting an QTreeWidgetItem will remove the item from the tree (this is safe at any time)
  // "delete this" is not good code but allowed and works here!
  delete this; // should be the last action
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
}

void Group::reloadFileSlotIfNewer() {
  xmlFileInfo->refresh();
  h5FileInfo->refresh();
  if(xmlFileInfo->lastModified()>xmlLastModified && h5FileInfo->lastModified()>h5LastModified) {
    xmlLastModified=xmlFileInfo->lastModified();
    h5LastModified=h5FileInfo->lastModified();
    reloadFileSlot();
  }
}
