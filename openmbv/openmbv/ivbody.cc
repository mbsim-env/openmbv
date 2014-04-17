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
#include "ivbody.h"
#include <Inventor/nodes/SoFile.h>
#include <Inventor/nodes/SoDrawStyle.h>

#include <Inventor/fields/SoMFColor.h>
#include <Inventor/actions/SoWriteAction.h>
#include <Inventor/actions/SoGetBoundingBoxAction.h>

#include <vector>
#include "edgecalculation.h"
#include "mainwindow.h"
#include "openmbvcppinterface/ivbody.h"
#include "openmbvcppinterface/group.h"

#include <QtConcurrentRun>
#include <QMenu>

using namespace std;

namespace OpenMBVGUI {

IvBody::IvBody(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind), calculateEdgesThread(this) {
  ivb=(OpenMBV::IvBody*)obj;
  iconFile="ivbody.svg";
  setIcon(0, Utils::QIconCached(iconFile));
  edgeCalc=NULL;

  // read XML
  string fileName=ivb->getIvFileName();
  // fix relative path name of file to be included (will hopefully work also on windows)
  fileName=boost::filesystem::absolute(fileName, boost::filesystem::path(ivb->getSeparateGroup()->getFileName()).parent_path()).string();

  // create so
  SoSeparator *sep=new SoSeparator; // to enable caching
  soSepRigidBody->addChild(sep);
  SoGroup *soIv=Utils::SoDBreadAllCached(fileName.c_str());
  sep->addChild(soIv);
  // connect object OpenMBVIvBodyMaterial in file to hdf5 mat if it is of type SoMaterial
  SoBase *ref=SoNode::getByName("OpenMBVIvBodyMaterial");
  if(ref && ref->getTypeId()==SoMaterial::getClassTypeId()) {
    ((SoMaterial*)ref)->diffuseColor.connectFrom(&mat->diffuseColor);
    ((SoMaterial*)ref)->specularColor.connectFrom(&mat->specularColor);
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
    // pre calculate edges, calculate crease edges and boundary edges in thread and call addEdgesToScene if finished
    connect(&calculateEdgesThread, SIGNAL(finished()), this, SLOT(addEdgesToScene()));
    calculateEdgesThread.start(QThread::IdlePriority);
  }

  connect(this, SIGNAL(statusBarShowMessage(const QString &, int)),
          MainWindow::getInstance()->statusBar(), SLOT(showMessage(const QString &, int)));
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
    creaseEdgesEditor->setSuffix(QString::fromUtf8("\xc2\xb0")); // utf8 degree sign
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

void IvBody::calculateEdges(string fullName, double creaseEdges, bool boundaryEdges) {
  // NOTE: It is not allowed here to use any variables of OpenMBV::IvBody since this class may aleady be
  // delete by a destructor call of a parent object of this object.
  // (OpenMBV::~Group deletes all children)
  QString str("Started edge calculation for %1 in a thread:"); str=str.arg(fullName.c_str());
  emit statusBarShowMessage(str, 1000);
  cout<<str.toStdString()<<endl;
  edgeCalc->preproces(fullName, true);
  if(creaseEdges>=0) edgeCalc->calcCreaseEdges(creaseEdges);
  if(boundaryEdges) edgeCalc->calcBoundaryEdges();
}

void IvBody::addEdgesToScene() {
  OpenMBV::IvBody *ivb=(OpenMBV::IvBody*)object;
  soOutLineSep->addChild(edgeCalc->getCoordinates());
  if(ivb->getCreaseEdges()>=0) soOutLineSep->addChild(edgeCalc->getCreaseEdges());
  if(ivb->getBoundaryEdges()) soOutLineSep->addChild(edgeCalc->getBoundaryEdges());
  QString str("Finished edge calculation for %1 and added to scene."); str=str.arg(ivb->getFullName().c_str());
  emit statusBarShowMessage(str, 1000);
  cout<<str.toStdString()<<endl;
}

}
