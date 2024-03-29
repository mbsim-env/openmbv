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
#include "rigidbody.h"
#include "mainwindow.h"
#include "compoundrigidbody.h"
#include <Inventor/nodes/SoScale.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoAntiSquish.h>
#include <Inventor/nodes/SoRotation.h>
#include <Inventor/nodes/SoMatrixTransform.h>
#include <Inventor/actions/SoGetMatrixAction.h>
#include <QMenu>
#include "utils.h"
#include "openmbvcppinterface/rigidbody.h"
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

RigidBody::RigidBody(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem_, SoGroup *soParent, int ind) : DynamicColoredBody(obj, parentItem_, soParent, ind) {
  rigidBody=std::static_pointer_cast<OpenMBV::RigidBody>(obj);
  //h5 dataset
  int rows=rigidBody->getRows();
  double dt;
  if(rows>=2) dt=rigidBody->getRow(1)[0]-rigidBody->getRow(0)[0]; else dt=0;
  resetAnimRange(rows, dt);

  // create so

  // path
  soPathSwitch=new SoSwitch;
  soSep->addChild(soPathSwitch);
  auto *pathSep=new SoSeparator;
  soPathSwitch->addChild(pathSep);
  auto *col=new SoBaseColor;
  col->rgb.setValue(0, 1, 0);
  pathSep->addChild(col);
  pathCoord=new SoCoordinate3;
  pathSep->addChild(pathCoord);
  pathLine=new SoLineSet;
  pathSep->addChild(pathLine);
  soPathSwitch->whichChild.setValue(rigidBody->getPath()?SO_SWITCH_ALL:SO_SWITCH_NONE);
  pathMaxFrameRead=-1;
  
  // translation (from hdf5)
  translation=new SoTranslation;
  soSep->addChild(translation);
  
  // rotation (from hdf5)
  rotationAlpha=new SoRotationXYZ;
  rotationAlpha->axis=SoRotationXYZ::X;
  soSep->addChild(rotationAlpha);
  rotationBeta=new SoRotationXYZ;
  rotationBeta->axis=SoRotationXYZ::Y;
  soSep->addChild(rotationBeta);
  rotationGamma=new SoRotationXYZ;
  rotationGamma->axis=SoRotationXYZ::Z;
  soSep->addChild(rotationGamma);
  rotation=new SoRotation;
  rotation->ref(); // do not add to scene graph (only for convinience)

  // till now the scene graph should be static (except color). So add a SoSeparator for caching
  soSepRigidBody=new SoSeparator;
  soSep->addChild(soSepRigidBody);

  // reference frame
  soReferenceFrameSwitch=new SoSwitch;
  soSepRigidBody->addChild(soReferenceFrameSwitch);
  soReferenceFrameSwitch->addChild(Utils::soFrame(1,1,false,refFrameScale));
  refFrameScale->ref();
  soReferenceFrameSwitch->whichChild.setValue(rigidBody->getReferenceFrame()?SO_SWITCH_ALL:SO_SWITCH_NONE);

  // add a group for the initial translation/rotation here (the SoTranslation/SoRotation is added later by InitialTransRotEditor)
  initTransRotGroup=new SoGroup;
  soSepRigidBody->addChild(initTransRotGroup);
  auto *initTrans=new SoTranslation;
  initTransRotGroup->addChild(initTrans);
  std::vector<double> t=rigidBody->getInitialTranslation();
  initTrans->translation.setValue(t[0], t[1], t[2]);
  auto *initRot=new SoRotation;
  initTransRotGroup->addChild(initRot);
  std::vector<double> r=rigidBody->getInitialRotation();
  initRot->rotation=Utils::cardan2Rotation(SbVec3f(r[0], r[1], r[2])).invert();

  // local frame
  soLocalFrameSwitch=new SoSwitch;
  soSepRigidBody->addChild(soLocalFrameSwitch);
  soLocalFrameSwitch->addChild(Utils::soFrame(1,1,false,localFrameScale));
  localFrameScale->ref();
  soLocalFrameSwitch->whichChild.setValue(rigidBody->getLocalFrame()?SO_SWITCH_ALL:SO_SWITCH_NONE);

  // initial scale
  auto *scale=new SoScale;
  scale->scaleFactor.setValue(rigidBody->getScaleFactor(),rigidBody->getScaleFactor(),rigidBody->getScaleFactor());
  soSepRigidBody->addChild(scale);

  if(clone && clone->properties)
    initialTransRotEditor=static_cast<RigidBody*>(clone)->initialTransRotEditor;
  else
    initialTransRotEditor=nullptr;
}

void RigidBody::createProperties() {
  DynamicColoredBody::createProperties();

  // GUI
  auto *moveCameraWith=new QAction(Utils::QIconCached("camerabody.svg"),"Move camera with this body", properties);
  connect(moveCameraWith,&QAction::triggered,this,[this](){
    auto *rigidBody=static_cast<RigidBody*>(properties->getParentObject());
    CompoundRigidBody *compoundRigidBodyParent;
    while((compoundRigidBodyParent=dynamic_cast<CompoundRigidBody*>(rigidBody->QTreeWidgetItem::parent()))!=nullptr) {
      rigidBody=compoundRigidBodyParent;
    }
    rigidBody->moveCameraWithSlot();
  });
  properties->addContextAction(moveCameraWith);

  // GUI editors
  if(!clone) {
    auto *localFrameEditor=new BoolEditor(properties, Utils::QIconCached("localframe.svg"), "Draw local frame", "RigidBody::localFrame");
    localFrameEditor->setOpenMBVParameter(rigidBody, &OpenMBV::RigidBody::getLocalFrame, &OpenMBV::RigidBody::setLocalFrame);
    properties->addPropertyAction(localFrameEditor->getAction());

    auto *referenceFrameEditor=new BoolEditor(properties, Utils::QIconCached("referenceframe.svg"), "Draw reference frame", "RigidBody::referenceFrame");
    referenceFrameEditor->setOpenMBVParameter(rigidBody, &OpenMBV::RigidBody::getReferenceFrame, &OpenMBV::RigidBody::setReferenceFrame);
    properties->addPropertyAction(referenceFrameEditor->getAction());

    auto *pathEditor=new BoolEditor(properties, Utils::QIconCached("path.svg"), "Draw path of reference frame", "RigidBody::paht");
    pathEditor->setOpenMBVParameter(rigidBody, &OpenMBV::RigidBody::getPath, &OpenMBV::RigidBody::setPath);
    properties->addPropertyAction(pathEditor->getAction());

    auto *scaleFactorEditor=new FloatEditor(properties, QIcon(), "Scaling");
    scaleFactorEditor->setOpenMBVParameter(rigidBody, &OpenMBV::RigidBody::getScaleFactor, &OpenMBV::RigidBody::setScaleFactor);

    // initial translation/rotation editor/dragger
    initialTransRotEditor=new TransRotEditor(properties, QIcon(), "Intial");
    initialTransRotEditor->setOpenMBVParameter(rigidBody, &OpenMBV::RigidBody::getInitialTranslation, &OpenMBV::RigidBody::setInitialTranslation,
                                                          &OpenMBV::RigidBody::getInitialRotation, &OpenMBV::RigidBody::setInitialRotation,
                                                          &OpenMBV::RigidBody::getDragger, &OpenMBV::RigidBody::setDragger);
  }
  initialTransRotEditor->setGroupMembers(initTransRotGroup);
}

RigidBody::~RigidBody() {
  rotation->unref();
  refFrameScale->unref();
  localFrameScale->unref();
}

void RigidBody::moveCameraWithSlot() {
  MainWindow::getInstance()->moveCameraWith(&translation->translation, &rotation->rotation);
}

double RigidBody::update() {
  if(rigidBody->getRows()==0) return 0; // do nothing for environement objects

  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()->getValue();
  vector<double> data=rigidBody->getRow(frame);
  
  // set scene values
  translation->translation.setValue(data[1], data[2], data[3]);
  rotationAlpha->angle.setValue(data[4]);
  rotationBeta->angle.setValue(data[5]);
  rotationGamma->angle.setValue(data[6]);
  rotation->rotation.setValue(Utils::cardan2Rotation(SbVec3f(data[4],data[5],data[6])).inverse()); // set rotation matrix (needed for move camera with body)

  // do not change "mat" if color has not changed to prevent
  // invalidating the render cache of the geometry.
  if(diffuseColor[0]<0) setColor(data[7]);

  // path
  if(rigidBody->getPath()) {
    for(int i=pathMaxFrameRead+1; i<=frame; i++) {
      vector<double> data=rigidBody->getRow(i);
      pathCoord->point.set1Value(i, data[1], data[2], data[3]);
    }
    pathMaxFrameRead=frame;
    pathLine->numVertices.setValue(1+frame);
  }

  return data[0];
}

QString RigidBody::getInfo() {
  float x, y, z;
  translation->translation.getValue().getValue(x,y,z);
  return DynamicColoredBody::getInfo()+
         QString("<hr width=\"10000\"/>")+
         QString("<b>Position:</b> %1, %2, %3<br/>").arg(x).arg(y).arg(z)+
         QString("<b>Rotation:</b> %1, %2, %3<br/>").arg(rotationAlpha->angle.getValue())
                                                    .arg(rotationBeta->angle.getValue())
                                                    .arg(rotationGamma->angle.getValue())+
         QString("<b>Rotation:</b> %1&deg;, %2&deg;, %3&deg;").arg(rotationAlpha->angle.getValue()*180/M_PI)
                                                    .arg(rotationBeta->angle.getValue()*180/M_PI)
                                                    .arg(rotationGamma->angle.getValue()*180/M_PI);
}

}
