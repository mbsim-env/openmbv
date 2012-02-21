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
#include "rigidbody.h"
#include "mainwindow.h"
#include "compoundrigidbody.h"
#include <Inventor/nodes/SoScale.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoAntiSquish.h>
#include <Inventor/nodes/SoRotation.h>
#include <Inventor/nodes/SoMatrixTransform.h>
#include <Inventor/actions/SoGetMatrixAction.h>
#include <QtGui/QMenu>
#include "utils.h"
#include "openmbvcppinterface/rigidbody.h"

using namespace std;

RigidBody::RigidBody(OpenMBV::Object *obj, QTreeWidgetItem *parentItem_, SoGroup *soParent, int ind) : DynamicColoredBody(obj, parentItem_, soParent, ind) {
  rigidBody=(OpenMBV::RigidBody*)obj;
  parentItem=parentItem_;
  //h5 dataset
  if(rigidBody->getParent()) { // do nothing for rigidbodies inside a compoundrigidbody
    int rows=rigidBody->getRows();
    double dt;
    if(rows>=2) dt=rigidBody->getRow(1)[0]-rigidBody->getRow(0)[0]; else dt=0;
    resetAnimRange(rows, dt);
  }

  // create so

  if(dynamic_cast<CompoundRigidBody*>(parentItem)==0) {
    // path
    soPathSwitch=new SoSwitch;
    soSep->addChild(soPathSwitch);
    SoSeparator *pathSep=new SoSeparator;
    soPathSwitch->addChild(pathSep);
    SoBaseColor *col=new SoBaseColor;
    col->rgb.setValue(0, 1, 0);
    pathSep->addChild(col);
    pathCoord=new SoCoordinate3;
    pathSep->addChild(pathCoord);
    pathLine=new SoLineSet;
    pathSep->addChild(pathLine);
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
  }
  else { // a dummmy refFrameScale
    refFrameScale=new SoScale;
    refFrameScale->ref();
    soSepRigidBody=soSep;
  }

  SoGroup *grp=new SoGroup;
  soSepRigidBody->addChild(grp);

  // initial translation/rotation editor/dragger
  initTrans=new SoTranslation;
  initRot=new SoRotation;
  transRotEditor=new TransRotEditor(this, QIcon(), "Intial Translation/Rotation", &initTrans->translation, &initRot->rotation);
  transRotEditor->setOpenMBVParameter(rigidBody, &OpenMBV::RigidBody::getInitialTranslation, &OpenMBV::RigidBody::setInitialTranslation,
                                                 &OpenMBV::RigidBody::getInitialRotation, &OpenMBV::RigidBody::setInitialRotation);
  transRotEditor->setDragger(grp);
  grp->addChild(initTrans);
  grp->addChild(initRot);

  if(dynamic_cast<CompoundRigidBody*>(parentItem)==0) {
    // local frame
    soLocalFrameSwitch=new SoSwitch;
    soSepRigidBody->addChild(soLocalFrameSwitch);
    soLocalFrameSwitch->addChild(Utils::soFrame(1,1,false,localFrameScale));
    localFrameScale->ref();
  
    // mat (from hdf5)
    mat=new SoMaterial;
    setColor(mat, 0);
    soSepRigidBody->addChild(mat);
    mat->shininess.setValue(0.9);
    if(!isnan(staticColor)) setColor(mat, staticColor);
  }
  else { // a dummmy localFrameScale
    localFrameScale=new SoScale;
    localFrameScale->ref();
    mat=static_cast<CompoundRigidBody*>(parentItem)->mat;
  }

  // initial scale
  SoScale *scale=new SoScale;
  scale->scaleFactor.setValue(rigidBody->getScaleFactor(),rigidBody->getScaleFactor(),rigidBody->getScaleFactor());
  soSepRigidBody->addChild(scale);

  if(dynamic_cast<CompoundRigidBody*>(parentItem)==0) {
    // GUI
    localFrameEditor=new BoolEditor(this, Utils::QIconCached(":/localframe.svg"), "Draw Local Frame", &soLocalFrameSwitch->whichChild);
    localFrameEditor->setOpenMBVParameter(rigidBody, &OpenMBV::RigidBody::getLocalFrame, &OpenMBV::RigidBody::setLocalFrame);
    referenceFrame=new BoolEditor(this, Utils::QIconCached(":/referenceframe.svg"),"Draw Reference Frame", &soReferenceFrameSwitch->whichChild);
    referenceFrame->setOpenMBVParameter(rigidBody, &OpenMBV::RigidBody::getReferenceFrame, &OpenMBV::RigidBody::setReferenceFrame);
    path=new BoolEditor(this, Utils::QIconCached(":/path.svg"),"Draw Path of Reference Frame", &soPathSwitch->whichChild);
    path->setOpenMBVParameter(rigidBody, &OpenMBV::RigidBody::getPath, &OpenMBV::RigidBody::setPath);
    connect(path->getAction(),SIGNAL(changed()),this,SLOT(update())); // special action required by path
    moveCameraWith=new QAction(Utils::QIconCached(":/camerabody.svg"),"Move Camera With This Body",this);
    moveCameraWith->setObjectName("RigidBody::moveCameraWith");
    connect(moveCameraWith,SIGNAL(triggered()),this,SLOT(moveCameraWithSlot()));
  }
}

RigidBody::~RigidBody() {
  if(dynamic_cast<CompoundRigidBody*>(parentItem)==0)
    rotation->unref();
  refFrameScale->unref();
  localFrameScale->unref();
}

QMenu* RigidBody::createMenu() {
  QMenu* menu=DynamicColoredBody::createMenu();
  menu->addSeparator()->setText("Properties from: RigidBody");
  menu->addAction(localFrameEditor->getAction());
  menu->addAction(referenceFrame->getAction());
  menu->addSeparator();
  menu->addAction(path->getAction());
  menu->addAction(transRotEditor->getAction());
  menu->addAction(transRotEditor->getDraggerAction());
  menu->addAction(moveCameraWith);
  return menu;
}

void RigidBody::moveCameraWithSlot() {
  MainWindow::getInstance()->moveCameraWith(&translation->translation, &rotation->rotation);
}

double RigidBody::update() {
  if(rigidBody->getRows()==-1) return 0; // do nothing for environement objects

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
  if(isnan(staticColor)) setColor(mat, data[7]);

  // path
  if(path->getAction()->isChecked()) {
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
