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

Group::Group(OpenMBV::Object* obj, QTreeWidgetItem *parentItem, SoGroup *soParent) : Object(obj, parentItem, soParent) {
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
    ObjectFactory(child[i], this, soSep);
  }

  // hide groups without childs
  if(childCount()==0) setHidden(true);

  // GUI
  saveFile=new QAction(Utils::QIconCached(":/savefile.svg"),"Save XML-File", this);
  saveFile->setObjectName("Group::saveFile");
  connect(saveFile,SIGNAL(activated()),this,SLOT(saveFileSlot()));
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
    grp->writeXML();
}
