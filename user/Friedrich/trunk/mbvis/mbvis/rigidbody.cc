#include "rigidbody.h"
#include <Inventor/nodes/SoScale.h>
#include <QtGui/QMenu>

using namespace std;

RigidBody::RigidBody(TiXmlElement *element) : Body(element) {
  // read XML
  TiXmlElement *e=element->FirstChildElement(MBVISNS"initialTranslation");
  vector<double> initTransValue=toVector(e->GetText());
  e=e->NextSiblingElement();
  vector<double> initRotValue=toVector(e->GetText());
  e=e->NextSiblingElement();
  double scaleValue=toVector(e->GetText())[0];

  // create so
  // initial rotation
  SoRotationXYZ *initRot;
  initRot=new SoRotationXYZ;
  initRot->axis=SoRotationXYZ::X;
  initRot->angle.setValue(initRotValue[0]);
  soSep->addChild(initRot);
  initRot=new SoRotationXYZ;
  initRot->axis=SoRotationXYZ::Y;
  initRot->angle.setValue(initRotValue[1]);
  soSep->addChild(initRot);
  initRot=new SoRotationXYZ;
  initRot->axis=SoRotationXYZ::Z;
  initRot->angle.setValue(initRotValue[2]);
  soSep->addChild(initRot);

  // initial translation
  SoTranslation *initTrans=new SoTranslation;
  initTrans->translation.setValue(initTransValue[0],initTransValue[1],initTransValue[2]);
  soSep->addChild(initTrans);

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

  // translation (from hdf5)
  translation=new SoTranslation;
  soSep->addChild(translation);

  // color (from hdf5)
  color=new SoBaseColor;
  soSep->addChild(color);

  // initial scale
  SoScale *scale=new SoScale;
  scale->scaleFactor.setValue(scaleValue,scaleValue,scaleValue);
  soSep->addChild(scale);

  // local frame
  soLocalFrameSwitch=new SoSwitch;
  soSep->addChild(soLocalFrameSwitch);
  soLocalFrameSwitch->addChild(soFrame(1,0));
  soLocalFrameSwitch->whichChild.setValue(SO_SWITCH_NONE);

  // reference frame
  soReferenceFrameSwitch=new SoSwitch;
  soSep->addChild(soReferenceFrameSwitch);
  soReferenceFrameSwitch->addChild(soFrame(1,0));
  soReferenceFrameSwitch->whichChild.setValue(SO_SWITCH_NONE);

  // GUI
  localFrame=new QAction("Draw Local Frame", 0);
  localFrame->setCheckable(true);
  connect(localFrame,SIGNAL(changed()),this,SLOT(localFrameSlot()));
  referenceFrame=new QAction("Draw Reference Frame", 0);
  referenceFrame->setCheckable(true);
  connect(referenceFrame,SIGNAL(changed()),this,SLOT(referenceFrameSlot()));
}

QMenu* RigidBody::createMenu() {
printf("CHANGED\n");
frame->setValue(5);
  QMenu* menu=Body::createMenu();
  menu->addSeparator();
  menu->addAction(localFrame);
  menu->addAction(referenceFrame);
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

void RigidBody::update() {
  // read from hdf5
  printf("UPDATE %d\n", frame->getValue());
  translation->translation.setValue(0, 0, 0);
  rotationAlpha->angle.setValue(0);
  rotationBeta->angle.setValue(0);
  rotationGamma->angle.setValue(0);
  color->rgb.setValue(1,0,0);
}
