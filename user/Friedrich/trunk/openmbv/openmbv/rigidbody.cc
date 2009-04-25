#include "config.h"
#include "rigidbody.h"
#include "mainwindow.h"
#include <Inventor/nodes/SoScale.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <QtGui/QMenu>

using namespace std;

RigidBody::RigidBody(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : Body(element, h5Parent, parentItem, soParent) {
  //h5 dataset
  h5Data=new H5::VectorSerie<double>;
  if(h5Group) {
    h5Data->open(*h5Group, "data");
    int rows=h5Data->getRows();
    double dt;
    if(rows>=2) dt=h5Data->getRow(1)[0]-h5Data->getRow(0)[0]; else dt=0;
    resetAnimRange(rows, dt);
  }
  
  // read XML
  TiXmlElement *e=element->FirstChildElement(OPENMBVNS"initialTranslation");
  vector<double> initTransValue=toVector(e->GetText());
  e=e->NextSiblingElement();
  vector<double> initRotValue=toVector(e->GetText());
  e=e->NextSiblingElement();
  double scaleValue=toVector(e->GetText())[0];

  // create so

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
  soReferenceFrameSwitch->addChild(soFrame(1,1,refFrameScale));
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
  soLocalFrameSwitch->addChild(soFrame(1,1,localFrameScale));
  soLocalFrameSwitch->whichChild.setValue(SO_SWITCH_NONE);

  // mat (from hdf5)
  mat=new SoMaterial;
  soSep->addChild(mat);
  mat->shininess.setValue(0.9);

  // initial scale
  SoScale *scale=new SoScale;
  scale->scaleFactor.setValue(scaleValue,scaleValue,scaleValue);
  soSep->addChild(scale);

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
}

QMenu* RigidBody::createMenu() {
  QMenu* menu=Body::createMenu();
  menu->addSeparator()->setText("Properties from: RigidBody");
  menu->addAction(localFrame);
  menu->addAction(referenceFrame);
  menu->addSeparator();
  menu->addAction(path);
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
  mat->diffuseColor.setHSVValue((1-data[7])*2/3,1,1);
  mat->specularColor.setHSVValue((1-data[7])*2/3,0.7,1);

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
