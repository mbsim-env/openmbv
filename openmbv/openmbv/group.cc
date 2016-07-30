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
#include "openmbvcppinterface/objectfactory.h"
#include "openmbvcppinterface/arrow.h"
#include "openmbvcppinterface/coilspring.h"
#include "openmbvcppinterface/nurbsdisk.h"
#include "openmbvcppinterface/indexedfaceset.h"
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
#include <mbxmlutilshelper/last_write_time.h>

using namespace std;

namespace OpenMBVGUI {

Group::Group(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : Object(obj, parentItem, soParent, ind) {
  grp=std::static_pointer_cast<OpenMBV::Group>(obj);
  iconFile="group.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // expand or collapse
  setExpanded(grp->getExpand());

  // seperator file? => new sub file
  if(grp->getSeparateFile()) {
    iconFile="h5file.svg";
    setIcon(0, Utils::QIconCached(iconFile));
    setText(0, grp->getFileName().c_str());
    setFlags(flags() & ~Qt::ItemIsEditable);
  }
  // read XML
  vector<std::shared_ptr<OpenMBV::Object> > child=grp->getObjects();
  for(unsigned int i=0; i<child.size(); i++) {
    if(child[i]->getClassName()=="Group" && (std::static_pointer_cast<OpenMBV::Group>(child[i]))->getObjects().size()==0) continue; // a hack for openmbvdeleterows.sh
    ObjectFactory::create(child[i], this, soSep, -1);
  }

  // timer for reloading file automatically
  reloadTimer=NULL;
  // if reloading is enabled and this Group is a toplevel file create timer
  std::shared_ptr<OpenMBV::Group> p=obj->getParent().lock();
  if(!p && MainWindow::getInstance()->getReloadTimeout()>0) {
    xmlLastModified=boost::myfilesystem::last_write_time(text(0).toStdString().c_str());
    h5LastModified =boost::myfilesystem::last_write_time((text(0).remove(text(0).count()-3, 3)+"h5").toStdString().c_str());

    reloadTimer=new QTimer(this);
    connect(reloadTimer,SIGNAL(timeout()),this,SLOT(reloadFileSlotIfNewer()));
    reloadTimer->start(MainWindow::getInstance()->getReloadTimeout());
  }
}

void Group::createProperties() {
  Object::createProperties();

  // GUI
  QAction *newObject=new QAction(Utils::QIconCached("newobject.svg"),"Create new Object", properties);
  connect(newObject,SIGNAL(triggered()),properties,SLOT(newObjectSlot()));
  properties->addContextAction(newObject);

  if(grp->getSeparateFile()) {
    QAction *saveFile=new QAction(Utils::QIconCached("savefile.svg"),"Save XML-file", properties);
    saveFile->setObjectName("Group::saveFile");
    connect(saveFile,SIGNAL(triggered()),properties,SLOT(saveFileSlot()));
    properties->addContextAction(saveFile);

    QAction *unloadFile=new QAction(Utils::QIconCached("unloadfile.svg"),"Unload XML/H5-file", properties);
    unloadFile->setObjectName("Group::unloadFile");
    connect(unloadFile,SIGNAL(triggered()),properties,SLOT(unloadFileSlot()));
    properties->addContextAction(unloadFile);

    QAction *reloadFile=new QAction(Utils::QIconCached("reloadfile.svg"),"Reload XML/H5-file", properties);
    reloadFile->setObjectName("Group::reloadFile");
    connect(reloadFile,SIGNAL(triggered()),properties,SLOT(reloadFileSlot()));
    properties->addContextAction(reloadFile);
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
  static vector<Utils::FactoryElement> factory={
    {Utils::FactoryElement(Utils::QIconCached("arrow.svg"),             "Arrow",             Utils::factory<OpenMBV::Arrow>())},
    {Utils::FactoryElement(Utils::QIconCached("coilspring.svg"),        "CoilSpring",        Utils::factory<OpenMBV::CoilSpring>())},
    {Utils::FactoryElement(Utils::QIconCached("invisiblebody.svg"),     "NurbsDisk",         Utils::factory<OpenMBV::NurbsDisk>())},
    {Utils::FactoryElement(Utils::QIconCached("indexedfaceset.svg"),    "IndexedFaceSet",    Utils::factory<OpenMBV::IndexedFaceSet>())},
    {Utils::FactoryElement(Utils::QIconCached("compoundrigidbody.svg"), "CompoundRigidBody", Utils::factory<OpenMBV::CompoundRigidBody>())},
    {Utils::FactoryElement(Utils::QIconCached("cube.svg"),              "Cube",              Utils::factory<OpenMBV::Cube>())},
    {Utils::FactoryElement(Utils::QIconCached("cuboid.svg"),            "Cuboid",            Utils::factory<OpenMBV::Cuboid>())},
    {Utils::FactoryElement(Utils::QIconCached("extrusion.svg"),         "Extrusion",         Utils::factory<OpenMBV::Extrusion>())},
    {Utils::FactoryElement(Utils::QIconCached("frame.svg"),             "Frame",             Utils::factory<OpenMBV::Frame>())},
    {Utils::FactoryElement(Utils::QIconCached("frustum.svg"),           "Frustum",           Utils::factory<OpenMBV::Frustum>())},
    {Utils::FactoryElement(Utils::QIconCached("invisiblebody.svg"),     "Grid",              Utils::factory<OpenMBV::Grid>())},
    {Utils::FactoryElement(Utils::QIconCached("invisiblebody.svg"),     "InvisibleBody",     Utils::factory<OpenMBV::InvisibleBody>())},
    {Utils::FactoryElement(Utils::QIconCached("ivbody.svg"),            "IvBody",            Utils::factory<OpenMBV::IvBody>())},
    {Utils::FactoryElement(Utils::QIconCached("rotation.svg"),          "Rotation",          Utils::factory<OpenMBV::Rotation>())},
    {Utils::FactoryElement(Utils::QIconCached("sphere.svg"),            "Sphere",            Utils::factory<OpenMBV::Sphere>())},
    {Utils::FactoryElement(Utils::QIconCached("invisiblebody.svg"),     "SpineExtrusion",    Utils::factory<OpenMBV::SpineExtrusion>())},
    {Utils::FactoryElement(Utils::QIconCached("path.svg"),              "Path",              Utils::factory<OpenMBV::Path>())},
    {Utils::FactoryElement(Utils::QIconCached("group.svg"),             "Group",             Utils::factory<OpenMBV::Group>())}
  };

  vector<string> existingNames;
  for(unsigned int j=0; j<grp->getObjects().size(); j++)
    existingNames.push_back(grp->getObjects()[j]->getName());

  std::shared_ptr<OpenMBV::Object> obj=Utils::createObjectEditor(factory, existingNames, "Create new Object");
  if(!obj) return;

  grp->addObject(obj);
  ObjectFactory::create(obj, this, soSep, -1);

  // apply object filter
  MainWindow::getInstance()->objectListFilter->applyFilter();
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
  // deleting an QTreeWidgetItem will remove the item from the tree (this is safe at any time)
  delete this;
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

}
