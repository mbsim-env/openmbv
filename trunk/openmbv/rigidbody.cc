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
#include <QtGui/QMenu>

using namespace std;

RigidBody::RigidBody(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : Body(element, h5Parent, parentItem, soParent) {
  if(h5Parent) {
    //h5 dataset
    h5Data=new H5::VectorSerie<double>;
    if(h5Group) {
      h5Data->open(*h5Group, "data");
      int rows=h5Data->getRows();
      double dt;
      if(rows>=2) dt=h5Data->getRow(1)[0]-h5Data->getRow(0)[0]; else dt=0;
      resetAnimRange(rows, dt);
    }
  }
  
  // read XML
  TiXmlElement *e=element->FirstChildElement(OPENMBVNS"minimalColorValue");
  if(e)
    minimalColorValue=toVector(e->GetText())[0];
  else
    minimalColorValue=0;
  e=element->FirstChildElement(OPENMBVNS"maximalColorValue");
  if(e)
    maximalColorValue=toVector(e->GetText())[0];
  else
    maximalColorValue=1;
  e=element->FirstChildElement(OPENMBVNS"staticColor");
  if(e)
    staticColor=atof(e->GetText());
  else
    staticColor=nan("");

  e=element->FirstChildElement(OPENMBVNS"initialTranslation");
  vector<double> initTransValue=toVector(e->GetText());
  e=e->NextSiblingElement();
  vector<double> initRotValue=toVector(e->GetText());
  e=e->NextSiblingElement();
  double scaleValue=toVector(e->GetText())[0];

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
    soPathSwitch->whichChild.setValue(SO_SWITCH_NONE);
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
  
    // reference frame
    soReferenceFrameSwitch=new SoSwitch;
    soSep->addChild(soReferenceFrameSwitch);
    soReferenceFrameSwitch->addChild(soFrame(1,1,false,refFrameScale));
    soReferenceFrameSwitch->whichChild.setValue(SO_SWITCH_NONE);
  }
  else { // a dummmy refFrameScale
    refFrameScale=new SoScale;
    refFrameScale->ref();
  }

  // dragger for initial translation and rotation
  SoGroup *grp=new SoGroup;
  soSep->addChild(grp);
  soDraggerSwitch=new SoSwitch;
  grp->addChild(soDraggerSwitch);
  soDraggerSwitch->whichChild.setValue(SO_SWITCH_NONE);
  SoCenterballDragger *soDragger=new SoCenterballDragger;
  soDraggerSwitch->addChild(soDragger);
  soDragger->addMotionCallback(draggerMoveCB, this);
  soDragger->addFinishCallback(draggerFinishCB, this);
  // initial translation from XML
  soDragger->center.setValue(initTransValue[0],initTransValue[1],initTransValue[2]);
  // initial rotation from XML
  soDragger->rotation.setValue(cardan2Rotation(SbVec3f(initRotValue[0],initRotValue[1],initRotValue[2])).invert());
  // scale of the dragger
  SoSurroundScale *draggerScale=new SoSurroundScale;
  draggerScale->setDoingTranslations(false);//TODO?
  draggerScale->numNodesUpToContainer.setValue(5);
  draggerScale->numNodesUpToReset.setValue(4);
  soDragger->setPart("surroundScale", draggerScale);

  // initial translation
  SoTranslation *initTrans=new SoTranslation;
  grp->addChild(initTrans);
  initTrans->translation.connectFrom(&soDragger->center);

  // initial rotation
  SoRotation *initRot=new SoRotation;
  grp->addChild(initRot);
  initRot->rotation.connectFrom(&soDragger->rotation);

  if(dynamic_cast<CompoundRigidBody*>(parentItem)==0) {
    // local frame
    soLocalFrameSwitch=new SoSwitch;
    soSep->addChild(soLocalFrameSwitch);
    soLocalFrameSwitch->addChild(soFrame(1,1,false,localFrameScale));
    soLocalFrameSwitch->whichChild.setValue(SO_SWITCH_NONE);
  
    // mat (from hdf5)
    mat=new SoMaterial;
    soSep->addChild(mat);
    mat->shininess.setValue(0.9);
    if(!isnan(staticColor)) {
      double m=1/(maximalColorValue-minimalColorValue);
      staticColor=m*staticColor-m*minimalColorValue;
      mat->diffuseColor.setHSVValue((1-staticColor)*2/3,1,1);
      mat->specularColor.setHSVValue((1-staticColor)*2/3,0.7,1);
    }
  }
  else { // a dummmy localFrameScale
    localFrameScale=new SoScale;
    localFrameScale->ref();
  }

  // initial scale
  SoScale *scale=new SoScale;
  scale->scaleFactor.setValue(scaleValue,scaleValue,scaleValue);
  soSep->addChild(scale);

  if(dynamic_cast<CompoundRigidBody*>(parentItem)==0) {
    // GUI
    localFrame=new QAction(QIcon(":/localframe.svg"),"Draw Local Frame", 0);
    localFrame->setCheckable(true);
    connect(localFrame,SIGNAL(changed()),this,SLOT(localFrameSlot()));
    referenceFrame=new QAction(QIcon(":/referenceframe.svg"),"Draw Reference Frame", 0);
    referenceFrame->setCheckable(true);
    connect(referenceFrame,SIGNAL(changed()),this,SLOT(referenceFrameSlot()));
    path=new QAction(QIcon(":/path.svg"),"Draw Path of Reference Frame", 0);
    path->setCheckable(true);
    connect(path,SIGNAL(changed()),this,SLOT(pathSlot()));
    dragger=new QAction(QIcon(":/centerballdragger.svg"),"Show Init. Trans./Rot. Dragger", 0);
    dragger->setCheckable(true);
    connect(dragger,SIGNAL(changed()),this,SLOT(draggerSlot()));
  }
}

QMenu* RigidBody::createMenu() {
  QMenu* menu=Body::createMenu();
  menu->addSeparator()->setText("Properties from: RigidBody");
  menu->addAction(localFrame);
  menu->addAction(referenceFrame);
  menu->addSeparator();
  menu->addAction(path);
  menu->addAction(dragger);
  return menu;
}

void RigidBody::localFrameSlot() {
  if(localFrame->isChecked())
    soLocalFrameSwitch->whichChild.setValue(SO_SWITCH_ALL);
  else
    soLocalFrameSwitch->whichChild.setValue(SO_SWITCH_NONE);
}

void RigidBody::referenceFrameSlot() {
  if(referenceFrame->isChecked())
    soReferenceFrameSwitch->whichChild.setValue(SO_SWITCH_ALL);
  else
    soReferenceFrameSwitch->whichChild.setValue(SO_SWITCH_NONE);
}

void RigidBody::pathSlot() {
  if(path->isChecked()) {
    soPathSwitch->whichChild.setValue(SO_SWITCH_ALL);
    update();
  }
  else
    soPathSwitch->whichChild.setValue(SO_SWITCH_NONE);
}

void RigidBody::draggerSlot() {
  if(dragger->isChecked())
    soDraggerSwitch->whichChild.setValue(SO_SWITCH_ALL);
  else
    soDraggerSwitch->whichChild.setValue(SO_SWITCH_NONE);
}

double RigidBody::update() {
  if(h5Group==0) return 0;
  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()->getValue();
  vector<double> data=h5Data->getRow(frame);
  
  // set scene values
  translation->translation.setValue(data[1], data[2], data[3]);
  rotationAlpha->angle.setValue(data[4]);
  rotationBeta->angle.setValue(data[5]);
  rotationGamma->angle.setValue(data[6]);
  if(isnan(staticColor)) {
    // norm color to [0,1] (from [minimalColorValue,maximalColorValue])
    double col=data[7];
    double m=1/(maximalColorValue-minimalColorValue);
    col=m*col-m*minimalColorValue;
    mat->diffuseColor.setHSVValue((1-col)*2/3,1,1);
    mat->specularColor.setHSVValue((1-col)*2/3,0.7,1);
  }

  // path
  if(path->isChecked()) {
    for(int i=pathMaxFrameRead+1; i<=frame; i++) {
      vector<double> data=h5Data->getRow(i);
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
  return Body::getInfo()+
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
  rotation2Cardan(dragger->rotation.getValue().inverse()).getValue(a,b,g);
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
  rotation2Cardan(dragger->rotation.getValue().inverse()).getValue(a,b,g);
  cout<<"New initial translation/rotation for: "<<me->getPath()<<endl
      <<"Translation: ["<<x<<", "<<y<<", "<<z<<"]"<<endl
      <<"Rotation: ["<<a<<", "<<b<<", "<<g<<"]"<<endl;
}