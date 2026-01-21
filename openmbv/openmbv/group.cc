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

#include "config.h"
#include "group.h"
#include <QMenu>
#include <QPushButton>
#include "objectfactory.h"
#include "mainwindow.h"
#include <string>
#include "utils.h"
#include <QMessageBox>
#include <QtCore/QFileInfo>
#include "openmbvcppinterface/objectfactory.h"
#include "openmbvcppinterface/arrow.h"
#include "openmbvcppinterface/coilspring.h"
#include "openmbvcppinterface/nurbsdisk.h"
#include "openmbvcppinterface/nurbscurve.h"
#include "openmbvcppinterface/nurbssurface.h"
#include "openmbvcppinterface/indexedlineset.h"
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
#include "openmbvcppinterface/ivscreenannotation.h"
#include "openmbvcppinterface/rotation.h"
#include "openmbvcppinterface/sphere.h"
#include "openmbvcppinterface/spineextrusion.h"
#include "openmbvcppinterface/cylindricalgear.h"
#include "openmbvcppinterface/cylinder.h"
#include "openmbvcppinterface/rack.h"
#include "openmbvcppinterface/bevelgear.h"
#include "openmbvcppinterface/planargear.h"
#include "openmbvcppinterface/path.h"
#include "openmbvcppinterface/group.h"

using namespace std;

namespace OpenMBVGUI {

Group::Group(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : Object(obj, parentItem, soParent, ind) {
  connect(this, &Group::reloadFileSignal, this, &Group::reloadFileSlot);
  connect(this, &Group::refreshFileSignal, this, &Group::refreshFileSlot);

  grp=std::static_pointer_cast<OpenMBV::Group>(obj);
  iconFile="group.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // expand or collapse
  setExpanded(grp->getExpand());

  // read XML
  vector<std::shared_ptr<OpenMBV::Object> > child=grp->getObjects();
  for(auto & i : child) {
    auto &iRef=*i;
    if(typeid(iRef)==typeid(OpenMBV::Group) && (std::static_pointer_cast<OpenMBV::Group>(i))->getObjects().empty()) continue; // a hack for openmbvdeleterows.sh
    ObjectFactory::create(i, this, soSep, -1);
  }
}

void Group::createProperties() {
  Object::createProperties();

  // GUI
  auto *newObject=new QAction(Utils::QIconCached("newobject.svg"),"Create new Object", properties);
  connect(newObject,&QAction::triggered,this,[this](){
    static_cast<Group*>(properties->getParentObject())->newObjectSlot();
  });
  properties->addContextAction(newObject);

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
    {Utils::FactoryElement(Utils::QIconCached("arrow.svg"),                "Arrow",             Utils::factory<OpenMBV::Arrow>())},
    {Utils::FactoryElement(Utils::QIconCached("coilspring.svg"),           "CoilSpring",        Utils::factory<OpenMBV::CoilSpring>())},
    {Utils::FactoryElement(Utils::QIconCached("invisiblebody.svg"),        "NurbsDisk",         Utils::factory<OpenMBV::NurbsDisk>())},
    {Utils::FactoryElement(Utils::QIconCached(""/*"nurbscurve.svg"*/),     "NurbsCurve",        Utils::factory<OpenMBV::NurbsCurve>())},
    {Utils::FactoryElement(Utils::QIconCached(""/*"nurbssurface.svg"*/),   "NurbsSurface",      Utils::factory<OpenMBV::NurbsSurface>())},
    {Utils::FactoryElement(Utils::QIconCached(""/*"indexedlineset.svg"*/), "IndexedLineSet",    Utils::factory<OpenMBV::IndexedLineSet>())},
    {Utils::FactoryElement(Utils::QIconCached(""/*"indexedfaceset.svg"*/), "IndexedFaceSet",    Utils::factory<OpenMBV::IndexedFaceSet>())},
    {Utils::FactoryElement(Utils::QIconCached("compoundrigidbody.svg"),    "CompoundRigidBody", Utils::factory<OpenMBV::CompoundRigidBody>())},
    {Utils::FactoryElement(Utils::QIconCached("cube.svg"),                 "Cube",              Utils::factory<OpenMBV::Cube>())},
    {Utils::FactoryElement(Utils::QIconCached("cuboid.svg"),               "Cuboid",            Utils::factory<OpenMBV::Cuboid>())},
    {Utils::FactoryElement(Utils::QIconCached("extrusion.svg"),            "Extrusion",         Utils::factory<OpenMBV::Extrusion>())},
    {Utils::FactoryElement(Utils::QIconCached("frame.svg"),                "Frame",             Utils::factory<OpenMBV::Frame>())},
    {Utils::FactoryElement(Utils::QIconCached("frustum.svg"),              "Frustum",           Utils::factory<OpenMBV::Frustum>())},
    {Utils::FactoryElement(Utils::QIconCached("invisiblebody.svg"),        "Grid",              Utils::factory<OpenMBV::Grid>())},
    {Utils::FactoryElement(Utils::QIconCached("invisiblebody.svg"),        "InvisibleBody",     Utils::factory<OpenMBV::InvisibleBody>())},
    {Utils::FactoryElement(Utils::QIconCached("ivbody.svg"),               "IvBody",            Utils::factory<OpenMBV::IvBody>())},
    {Utils::FactoryElement(Utils::QIconCached("ivscreenannotation.svg"),   "ivscreenannotation",Utils::factory<OpenMBV::IvScreenAnnotation>())},
    {Utils::FactoryElement(Utils::QIconCached("rotation.svg"),             "Rotation",          Utils::factory<OpenMBV::Rotation>())},
    {Utils::FactoryElement(Utils::QIconCached("sphere.svg"),               "Sphere",            Utils::factory<OpenMBV::Sphere>())},
    {Utils::FactoryElement(Utils::QIconCached("invisiblebody.svg"),        "SpineExtrusion",    Utils::factory<OpenMBV::SpineExtrusion>())},
    {Utils::FactoryElement(Utils::QIconCached(""/*"cylindricalgear.svg"*/),"CylindricalGear",   Utils::factory<OpenMBV::CylindricalGear>())},
    {Utils::FactoryElement(Utils::QIconCached("cylinder.svg"),             "Cylinder",          Utils::factory<OpenMBV::Cylinder>())},
    {Utils::FactoryElement(Utils::QIconCached(""/*"rack.svg"*/),           "Rack",              Utils::factory<OpenMBV::Rack>())},
    {Utils::FactoryElement(Utils::QIconCached(""/*"bevelgear.svg"*/),      "BevelGear",         Utils::factory<OpenMBV::BevelGear>())},
    {Utils::FactoryElement(Utils::QIconCached(""/*"planargear.svg"*/),     "PlanarGear",        Utils::factory<OpenMBV::PlanarGear>())},
    {Utils::FactoryElement(Utils::QIconCached("path.svg"),                 "Path",              Utils::factory<OpenMBV::Path>())},
    {Utils::FactoryElement(Utils::QIconCached("group.svg"),                "Group",             Utils::factory<OpenMBV::Group>())}
  };

  vector<string> existingNames;
  for(auto & j : grp->getObjects())
    existingNames.push_back(j->getName());

  std::shared_ptr<OpenMBV::Object> obj=Utils::createObjectEditor(factory, existingNames, "Create new Object");
  if(!obj) return;

  grp->addObject(obj);
  ObjectFactory::create(obj, this, soSep, -1);

  // apply object filter
  MainWindow::getInstance()->objectListFilter->applyFilter();
}

void Group::unloadFileSlot() {
  MainWindow::getInstance()->openMBVBodyForLastFrame.reset(); // just required if openMBVBodyForLastFrame stores a pointer to the here removed object
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
  MainWindow::getInstance()->openFile(fileName, parent, parent?((Group*)parent)->soSep:nullptr, ind);
  // set new item the current item: this selects and scroll to the new widget
  auto grp = parent ? parent->child(ind) : MainWindow::getInstance()->objectList->invisibleRootItem()->child(ind);
  MainWindow::getInstance()->objectList->setCurrentItem(grp, 0, QItemSelectionModel::NoUpdate);
  MainWindow::getInstance()->fileReloaded(static_cast<Group*>(grp));
}

void Group::refreshFileSlot() {
  grp->refresh();

  // if we are at the first frame we may need to redraw (refresh the scene) since the first frame may
  // also be the ALL NULL position.
  auto *mw=MainWindow::getInstance();
  if(mw->frameNode->index[0]==0 && object->getParent().expired()) // only needed for the root Group (which has no parent)
    mw->frameNode->index.setValue(0); // this calls a redraw of the scene
}

bool Group::requestFlush() {
  return grp->requestFlush();
}

}
