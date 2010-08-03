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
#include <Inventor/nodes/SoSurroundScale.h>
#include <Inventor/nodes/SoAntiSquish.h>
#include <Inventor/nodes/SoRotation.h>
#include <Inventor/nodes/SoMatrixTransform.h>
#include <Inventor/draggers/SoCenterballDragger.h>
#include <Inventor/actions/SoGetMatrixAction.h>
#include <QtGui/QMenu>
#include "utils.h"
#include "openmbvcppinterface/rigidbody.h"

using namespace std;

RigidBody::RigidBody(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent) : DynamicColoredBody(obj, parentItem, soParent) {
  rigidBody=(OpenMBV::RigidBody*)obj;
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
    soReferenceFrameSwitch->whichChild.setValue(rigidBody->getReferenceFrame()?SO_SWITCH_ALL:SO_SWITCH_NONE);
  }
  else { // a dummmy refFrameScale
    refFrameScale=new SoScale;
    refFrameScale->ref();
    soSepRigidBody=soSep;
  }

  // dragger for initial translation and rotation
  SoGroup *grp=new SoGroup;
  soSepRigidBody->addChild(grp);
  soDraggerSwitch=new SoSwitch;
  grp->addChild(soDraggerSwitch);
  soDraggerSwitch->whichChild.setValue(SO_SWITCH_NONE);
  // the dragger is added on the first click on the entry in the property menu (Because the constructor of the dragger takes a long time to execute (performace reason))
  soDragger=0;

  // initial translation
  initTrans=new SoTranslation;
  grp->addChild(initTrans);
  initTrans->translation.setValue(rigidBody->getInitialTranslation()[0],rigidBody->getInitialTranslation()[1],rigidBody->getInitialTranslation()[2]); // this is later changed to connectFrom if the dragger exists

  // initial rotation
  initRot=new SoRotation;
  grp->addChild(initRot);
  initRot->rotation.setValue(Utils::cardan2Rotation(SbVec3f(rigidBody->getInitialRotation()[0],rigidBody->getInitialRotation()[1],rigidBody->getInitialRotation()[2])).invert()); // this is later changed to connectFrom if the dragger exists

  if(dynamic_cast<CompoundRigidBody*>(parentItem)==0) {
    // local frame
    soLocalFrameSwitch=new SoSwitch;
    soSepRigidBody->addChild(soLocalFrameSwitch);
    soLocalFrameSwitch->addChild(Utils::soFrame(1,1,false,localFrameScale));
    soLocalFrameSwitch->whichChild.setValue(rigidBody->getLocalFrame()?SO_SWITCH_ALL:SO_SWITCH_NONE);
  
    // mat (from hdf5)
    mat=new SoMaterial;
    soSepRigidBody->addChild(mat);
    mat->shininess.setValue(0.9);
    if(!isnan(staticColor)) setColor(mat, staticColor);
  }
  else { // a dummmy localFrameScale
    localFrameScale=new SoScale;
    localFrameScale->ref();
  }

  // initial scale
  SoScale *scale=new SoScale;
  scale->scaleFactor.setValue(rigidBody->getScaleFactor(),rigidBody->getScaleFactor(),rigidBody->getScaleFactor());
  soSepRigidBody->addChild(scale);

  if(dynamic_cast<CompoundRigidBody*>(parentItem)==0) {
    // GUI
    localFrame=new QAction(Utils::QIconCached(":/localframe.svg"),"Draw Local Frame", this);
    localFrame->setCheckable(true);
    localFrame->setChecked(rigidBody->getLocalFrame());
    localFrame->setObjectName("RigidBody::localFrame");
    connect(localFrame,SIGNAL(changed()),this,SLOT(localFrameSlot()));
    referenceFrame=new QAction(Utils::QIconCached(":/referenceframe.svg"),"Draw Reference Frame", this);
    referenceFrame->setCheckable(true);
    referenceFrame->setChecked(rigidBody->getReferenceFrame());
    referenceFrame->setObjectName("RigidBody::referenceFrame");
    connect(referenceFrame,SIGNAL(changed()),this,SLOT(referenceFrameSlot()));
    path=new QAction(Utils::QIconCached(":/path.svg"),"Draw Path of Reference Frame", this);
    path->setCheckable(true);
    path->setChecked(rigidBody->getPath());
    path->setObjectName("RigidBody::path");
    connect(path,SIGNAL(changed()),this,SLOT(pathSlot()));
    dragger=new QAction(Utils::QIconCached(":/centerballdragger.svg"),"Show Init. Trans./Rot. Dragger", this);
    dragger->setCheckable(true);
    dragger->setObjectName("RigidBody::dragger");
    connect(dragger,SIGNAL(changed()),this,SLOT(draggerSlot()));
    moveCameraWith=new QAction(Utils::QIconCached(":/camerabody.svg"),"Move Camera With This Body",this);
    moveCameraWith->setObjectName("RigidBody::moveCameraWith");
    connect(moveCameraWith,SIGNAL(triggered()),this,SLOT(moveCameraWithSlot()));
  }
}

QMenu* RigidBody::createMenu() {
  QMenu* menu=DynamicColoredBody::createMenu();
  menu->addSeparator()->setText("Properties from: RigidBody");
  menu->addAction(localFrame);
  menu->addAction(referenceFrame);
  menu->addSeparator();
  menu->addAction(path);
  menu->addAction(dragger);
  menu->addAction(moveCameraWith);
  return menu;
}

void RigidBody::localFrameSlot() {
  if(localFrame->isChecked()) {
    soLocalFrameSwitch->whichChild.setValue(SO_SWITCH_ALL);
    rigidBody->setLocalFrame(true);
  }
  else {
    soLocalFrameSwitch->whichChild.setValue(SO_SWITCH_NONE);
    rigidBody->setLocalFrame(false);
  }
}

void RigidBody::referenceFrameSlot() {
  if(referenceFrame->isChecked()) {
    soReferenceFrameSwitch->whichChild.setValue(SO_SWITCH_ALL);
    rigidBody->setReferenceFrame(false);
  }
  else {
    soReferenceFrameSwitch->whichChild.setValue(SO_SWITCH_NONE);
    rigidBody->setReferenceFrame(false);
  }
}

void RigidBody::pathSlot() {
  if(path->isChecked()) {
    soPathSwitch->whichChild.setValue(SO_SWITCH_ALL);
    rigidBody->setPath(true);
    update();
  }
  else {
    soPathSwitch->whichChild.setValue(SO_SWITCH_NONE);
    rigidBody->setPath(false);
  }
}

void RigidBody::draggerSlot() {
  if(soDragger==0) {
    soDragger=new SoCenterballDragger;
    soDraggerSwitch->addChild(soDragger);
    soDragger->addMotionCallback(draggerMoveCB, this);
    soDragger->addFinishCallback(draggerFinishCB, this);
    // initial translation from XML
    soDragger->center=initTrans->translation;
    // initial rotation from XML
    soDragger->rotation=initRot->rotation;
    // scale of the dragger
    SoSurroundScale *draggerScale=new SoSurroundScale;
    draggerScale->setDoingTranslations(false);//TODO?
    draggerScale->numNodesUpToContainer.setValue(5);
    draggerScale->numNodesUpToReset.setValue(4);
    soDragger->setPart("surroundScale", draggerScale);

    initTrans->translation.connectFrom(&soDragger->center);
    initRot->rotation.connectFrom(&soDragger->rotation);
  }
  if(dragger->isChecked())
    soDraggerSwitch->whichChild.setValue(SO_SWITCH_ALL);
  else
    soDraggerSwitch->whichChild.setValue(SO_SWITCH_NONE);
}

void RigidBody::moveCameraWithSlot() {
  MainWindow::getInstance()->moveCameraWith(&translation->translation, &rotation->rotation);
}

double RigidBody::update() {
  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()->getValue();
  vector<double> data=rigidBody->getRow(frame);
  
  // set scene values
  translation->translation.setValue(data[1], data[2], data[3]);
  rotationAlpha->angle.setValue(data[4]);
  rotationBeta->angle.setValue(data[5]);
  rotationGamma->angle.setValue(data[6]);
  rotation->rotation.setValue(Utils::cardan2Rotation(SbVec3f(data[4],data[5],data[6])).inverse()); // set rotatoin matrix (needed for move camera with body)

  // do not change "mat" if color has not changed to prevent
  // invalidating the render cache of the geometry.
  if(isnan(staticColor)) setColor(mat, data[7]);

  // path
  if(path->isChecked()) {
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
         QString("-----<br/>")+
         QString("<b>Position:</b> %1, %2, %3<br/>").arg(x).arg(y).arg(z)+
         QString("<b>Rotation:</b> %1, %2, %3<br/>").arg(rotationAlpha->angle.getValue())
                                                    .arg(rotationBeta->angle.getValue())
                                                    .arg(rotationGamma->angle.getValue())+
         QString("<b>Rotation:</b> %1&deg;, %2&deg;, %3&deg;<br/>").arg(rotationAlpha->angle.getValue()*180/M_PI)
                                                    .arg(rotationBeta->angle.getValue()*180/M_PI)
                                                    .arg(rotationGamma->angle.getValue()*180/M_PI);
}

void RigidBody::draggerMoveCB(void *, SoDragger *dragger_) {
  SoCenterballDragger* dragger=(SoCenterballDragger*)dragger_;
  float x,y,z;
  dragger->center.getValue().getValue(x,y,z);
  float a, b, g;
  Utils::rotation2Cardan(dragger->rotation.getValue().inverse()).getValue(a,b,g);
  MainWindow::getInstance()->statusBar()->showMessage(QString("Trans: [%1, %2, %3]; Rot: [%4, %5, %6]").
    arg(x,0,'f',6).arg(y,0,'f',6).arg(z,0,'f',6).
    arg(a,0,'f',6).arg(b,0,'f',6).arg(g,0,'f',6));
}

void RigidBody::draggerFinishCB(void *me_, SoDragger *dragger_) {
  RigidBody* me=(RigidBody*)me_;
  SoCenterballDragger* dragger=(SoCenterballDragger*)dragger_;
  float x,y,z;
  dragger->center.getValue().getValue(x,y,z);
  float a, b, g;
  Utils::rotation2Cardan(dragger->rotation.getValue().inverse()).getValue(a,b,g);
  cout<<"New initial translation/rotation for: "<<me->getPath()<<endl
      <<"Translation: ["<<x<<", "<<y<<", "<<z<<"]"<<endl
      <<"Rotation: ["<<a<<", "<<b<<", "<<g<<"]"<<endl;
}
