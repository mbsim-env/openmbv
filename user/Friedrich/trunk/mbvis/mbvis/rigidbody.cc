#include "rigidbody.h"
#include <Inventor/nodes/SoScale.h>
#include <QtGui/QMenu>

using namespace std;

RigidBody::RigidBody(TiXmlElement *element, H5::Group *h5Parent) : Body(element, h5Parent) {
  //h5 dataset
  h5Data=new H5::VectorSerie<double>;
  h5Data->open(*h5Group, "data");
  
  // read XML
  TiXmlElement *e=element->FirstChildElement(MBVISNS"initialTranslation");
  vector<double> initTransValue=toVector(e->GetText());
  e=e->NextSiblingElement();
  vector<double> initRotValue=toVector(e->GetText());
  e=e->NextSiblingElement();
  double scaleValue=toVector(e->GetText())[0];

  // create so
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
  soReferenceFrameSwitch->addChild(soFrame(1,0));
  soReferenceFrameSwitch->whichChild.setValue(SO_SWITCH_NONE);

  // initial translation
  SoTranslation *initTrans=new SoTranslation;
  initTrans->translation.setValue(initTransValue[0],initTransValue[1],initTransValue[2]);
  soSep->addChild(initTrans);

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

  // local frame
  soLocalFrameSwitch=new SoSwitch;
  soSep->addChild(soLocalFrameSwitch);
  soLocalFrameSwitch->addChild(soFrame(1,0));
  soLocalFrameSwitch->whichChild.setValue(SO_SWITCH_NONE);

  // color (from hdf5)
  color=new SoBaseColor;
  soSep->addChild(color);

  // initial scale
  SoScale *scale=new SoScale;
  scale->scaleFactor.setValue(scaleValue,scaleValue,scaleValue);
  soSep->addChild(scale);

  // GUI
  localFrame=new QAction("Draw Local Frame", 0);
  localFrame->setCheckable(true);
  connect(localFrame,SIGNAL(changed()),this,SLOT(localFrameSlot()));
  referenceFrame=new QAction("Draw Reference Frame", 0);
  referenceFrame->setCheckable(true);
  connect(referenceFrame,SIGNAL(changed()),this,SLOT(referenceFrameSlot()));
}

QMenu* RigidBody::createMenu() {
  QMenu* menu=Body::createMenu();
  menu->addSeparator();
  QAction *type=new QAction("Properties from: RigidBody", menu);
  type->setEnabled(false);
  menu->addAction(type);
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
  vector<double> data=h5Data->getRow(frame->getValue());
  
  // set scene values
  translation->translation.setValue(data[1], data[2], data[3]);
  rotationAlpha->angle.setValue(data[4]);
  rotationBeta->angle.setValue(data[5]);
  rotationGamma->angle.setValue(data[6]);
  color->rgb.setHSVValue((1-data[7])*2/3,1,1);
  color->rgb.setHSVValue((1-1)*2/3,1,1);
}
