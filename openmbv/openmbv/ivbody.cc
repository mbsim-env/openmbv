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
#include "ivbody.h"
#include <Inventor/nodes/SoFile.h>
#include <Inventor/nodes/SoDrawStyle.h>

#include <Inventor/fields/SoMFColor.h>
#include <Inventor/actions/SoWriteAction.h>
#include <Inventor/actions/SoGetBoundingBoxAction.h>
#include <Inventor/actions/SoSearchAction.h>

#include <vector>
#include <boost/container_hash/hash.hpp>
#include "edgecalculation.h"
#include "mainwindow.h"
#include "openmbvcppinterface/ivbody.h"
#include "openmbvcppinterface/group.h"

#include <QMenu>

using namespace std;

namespace OpenMBVGUI {

IvBody::IvBody(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind), calculateEdgesThread(this) {
  ivb=std::static_pointer_cast<OpenMBV::IvBody>(obj);
  iconFile="ivbody.svg";
  setIcon(0, Utils::QIconCached(iconFile));
  edgeCalc=nullptr;

  // read XML
  string fileName=ivb->getIvFileName();

  // create so
  auto *sep=new SoSeparator; // to enable caching
  sep->renderCaching.setValue(SoSeparator::ON);
  sep->boundingBoxCaching.setValue(SoSeparator::ON);
  soSepRigidBody->addChild(sep);

  auto hashData = make_tuple(
    ivb->getRemoveNodesByName(),
    ivb->getRemoveNodesByType()
  );

  SoGroup *soIv;
  if(!fileName.empty())
    soIv=Utils::SoDBreadAllFileNameCached(fileName, boost::hash<decltype(hashData)>{}(hashData));
  else
    soIv=Utils::SoDBreadAllContentCached(ivb->getIvContent(), boost::hash<decltype(hashData)>{}(hashData));
  sep->addChild(soIv);
  if(!soIv)
    return;

  // search and remove specific nodes
  auto removeNode=[soIv, &fileName, this](const function<void(SoSearchAction &sa)> &find){
    SoSearchAction sa;
    sa.setInterest(SoSearchAction::ALL);
    find(sa);
    sa.apply(soIv);
    auto pl = sa.getPaths();
    for(int i=0; i<pl.getLength(); ++i) {
      msg(Info)<<"Removing the following node for IVBody from file '"<<fileName<<"':"<<endl;
      auto *p = pl[i];
      for(int j=1; j<p->getLength(); ++j) {
        auto *n = p->getNode(j);
        msg(Info)<<string(2*j, ' ')<<"- Name: '"<<n->getName()<<"'; Type: '"<<n->getTypeId().getName().getString()<<"'"<<endl;
      }
      static_cast<SoGroup*>(p->getNodeFromTail(1))->removeChild(p->getIndexFromTail(0));
    }
  };
  // remove nodes by name
  for(auto &name : ivb->getRemoveNodesByName())
    removeNode([&name](auto &sa){ sa.setName(name.c_str()); });
  // remove nodes by type
  for(auto &type : ivb->getRemoveNodesByType())
    removeNode([&type](auto &sa){ sa.setType(SoType::fromName(type.c_str())); });

  // connect object OpenMBVIvBodyMaterial in file to hdf5 mat if it is of type SoMaterial
  SoBase *ref=Utils::getChildNodeByName(soIv, "OpenMBVIvBodyMaterial");
  if(ref && ref->getTypeId()==SoMaterial::getClassTypeId()) {
    ((SoMaterial*)ref)->diffuseColor.connectFrom(&mat->diffuseColor);
    ((SoMaterial*)ref)->specularColor.connectFrom(&mat->specularColor);
    sep->renderCaching.setValue(SoSeparator::AUTO);
  }

  // scale ref/localFrame
  SoGetBoundingBoxAction bboxAction(SbViewportRegion(0,0));
  bboxAction.apply(soSepRigidBody);
  float x1,y1,z1,x2,y2,z2;
  bboxAction.getBoundingBox().getBounds(x1,y1,z1,x2,y2,z2);
  double size=min(x2-x1,min(y2-y1,z2-z1))*ivb->getScaleFactor();
  refFrameScale->scaleFactor.setValue(size,size,size);
  localFrameScale->scaleFactor.setValue(size,size,size);

  // outline
  if(ivb->getCreaseEdges()>=0 || ivb->getBoundaryEdges()) {
    soSepRigidBody->addChild(soOutLineSwitch);
    // get data for edge calculation from scene
    edgeCalc=new EdgeCalculation(soIv);
    // wait for this thread; see addEdgesToScene for erase (currently only used for --autoExit)
    MainWindow::getInstance()->waitFor.insert(&calculateEdgesThread);
    // pre calculate edges, calculate crease edges and boundary edges in thread and call addEdgesToScene if finished
    connect(&calculateEdgesThread, &CalculateEdgesThread::finished, this, &IvBody::addEdgesToScene);
    calculateEdgesThread.start(QThread::IdlePriority);
  }

  connect(this, &IvBody::statusBarShowMessage,
          MainWindow::getInstance()->statusBar(), &QStatusBar::showMessage);
}

void IvBody::createProperties() {
  RigidBody::createProperties();

  // GUI editors
  if(!clone) {
    properties->updateHeader();
    StringEditor *ivFileNameEditor=new StringEditor(properties, QIcon(), "IV file name");
    ivFileNameEditor->setOpenMBVParameter(ivb, &OpenMBV::IvBody::getIvFileName, &OpenMBV::IvBody::setIvFileName);

    FloatEditor *creaseEdgesEditor=new FloatEditor(properties, QIcon(), "Crease edges bound");
    creaseEdgesEditor->setRange(0, 180); // degree
    creaseEdgesEditor->setStep(5); // degree
    creaseEdgesEditor->setSuffix(QString::fromUtf8(R"(°)")); // utf8 degree sign
    creaseEdgesEditor->setFactor(M_PI/180); // degree to rad conversion factor
    creaseEdgesEditor->setOpenMBVParameter(ivb, &OpenMBV::IvBody::getCreaseEdges, &OpenMBV::IvBody::setCreaseEdges);

    BoolEditor *boundaryEdgesEditor=new BoolEditor(properties, QIcon(), "Draw boundary edges", "IvBody::drawBoundaryEdges");
    boundaryEdgesEditor->setOpenMBVParameter(ivb, &OpenMBV::IvBody::getBoundaryEdges, &OpenMBV::IvBody::setBoundaryEdges);
  }
}

IvBody::~IvBody() {
  calculateEdgesThread.wait(); // wait for thread to finish
  delete edgeCalc;
}

void IvBody::calculateEdges(const string& fullName, double creaseEdges, bool boundaryEdges) {
  // NOTE: It is not allowed here to use any variables of OpenMBV::IvBody since this class may aleady be
  // delete by a destructor call of a parent object of this object.
  // (OpenMBV::~Group deletes all children)
  QString str("Started edge calculation for %1 in a thread:"); str=str.arg(fullName.c_str());
  statusBarShowMessage(str, 1000);
  msg(Info)<<str.toStdString()<<endl;
  edgeCalc->preproces(fullName, true);
  if(creaseEdges>=0) edgeCalc->calcCreaseEdges(creaseEdges);
  if(boundaryEdges) edgeCalc->calcBoundaryEdges();
}

void IvBody::addEdgesToScene() {
  std::shared_ptr<OpenMBV::IvBody> ivb=std::static_pointer_cast<OpenMBV::IvBody>(object);
  soOutLineSep->renderCaching.setValue(SoSeparator::ON);
  soOutLineSep->boundingBoxCaching.setValue(SoSeparator::ON);
  soOutLineSep->addChild(edgeCalc->getCoordinates());
  if(ivb->getCreaseEdges()>=0) soOutLineSep->addChild(edgeCalc->getCreaseEdges());
  if(ivb->getBoundaryEdges()) soOutLineSep->addChild(edgeCalc->getBoundaryEdges());
  QString str("Finished edge calculation for %1 and added to scene."); str=str.arg(ivb->getFullName().c_str());
  statusBarShowMessage(str, 1000);
  msg(Info)<<str.toStdString()<<endl;
  MainWindow::getInstance()->waitFor.erase(&calculateEdgesThread);
}

}
