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
#include "openmbvcppinterface/group.h"
#include <string>
#include "utils.h"
#include <QtGui/QMessageBox>

using namespace std;

Group::Group(OpenMBV::Object* obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : Object(obj, parentItem, soParent, ind) {
  grp=(OpenMBV::Group*)obj;
  iconFile=":/group.svg";
  setIcon(0, Utils::QIconCached(iconFile.c_str()));

  // expand or collapse
  setExpanded(grp->getExpand());

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

  // GUI MFMF maybe also grp->getSeparateFile() is possible instead of grp->getParent()==NULL
  if(grp->getParent()==NULL) {
    saveFile=new QAction(Utils::QIconCached(":/savefile.svg"),"Save XML-file", this);
//MFMF multiedit    saveFile->setObjectName("Group::saveFile");
    connect(saveFile,SIGNAL(activated()),this,SLOT(saveFileSlot()));
    properties->addContextAction(saveFile);

    unloadFile=new QAction(Utils::QIconCached(":/unloadfile.svg"),"Unload XML/H5-file", this);
//MFMF multiedit    unloadFile->setObjectName("Group::unloadFile");
    connect(unloadFile,SIGNAL(activated()),this,SLOT(unloadFileSlot()));
    properties->addContextAction(unloadFile);

    reloadFile=new QAction(Utils::QIconCached(":/reloadfile.svg"),"Reload XML/H5-file", this);
//MFMF multiedit    reloadFile->setObjectName("Group::reloadFile");
    connect(reloadFile,SIGNAL(activated()),this,SLOT(reloadFileSlot()));
    properties->addContextAction(reloadFile);

    // timer for reloading file automatically
    reloadTimer=NULL;
    // if reloading is enabled and this Group is a toplevel file create timer
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

void Group::saveFileSlot() {
  if(QMessageBox::warning(0, "Overwrite XML-File", QString(
       "Save current object porperties in XML-File.\n"
       "\n"
       "Saving will overwrite the following files:\n"
       "- OpenMBV-XML-file '%1'\n"
       "- OpenMBV-Parameter-XML-file '%2'\n"
       "- all included OpenMBV-XML-Files\n"
       "- all dedicated OpenMBV-Parameter-XML-Files")
        .arg(grp->getFileName().c_str())
        .arg((grp->getFileName().substr(0,grp->getFileName().length()-4)+".param.xml").c_str()),
       QMessageBox::Save | QMessageBox::Cancel)==QMessageBox::Save)
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

string Group::getPath() {
  if(grp->getSeparateFile())
    return text(0).toStdString();
  return Object::getPath();
}
