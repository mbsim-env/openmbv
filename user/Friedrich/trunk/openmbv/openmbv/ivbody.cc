#include "config.h"
#include "ivbody.h"
#include "tinynamespace.h"
#include <Inventor/nodes/SoFile.h>
#include <Inventor/nodes/SoDrawStyle.h>

#include <Inventor/fields/SoMFColor.h>
#include <Inventor/actions/SoWriteAction.h>
#include <Inventor/actions/SoGetBoundingBoxAction.h>

#include <vector>

using namespace std;

IvBody::IvBody(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : RigidBody(element, h5Parent, parentItem, soParent) {
  iconFile=":/ivbody.svg";
  setIcon(0, QIcon(iconFile.c_str()));

  // read XML
  TiXmlElement *e=element->FirstChildElement(OPENMBVNS"ivFileName");
  string fileName=e->GetText();

  // fix relative path name of file to be included (will hopefully work also on windows)
  fileName=fixPath(e->GetDocument()->ValueStr(), fileName);

  // create so
  SoFile *file=new SoFile;
  soSep->addChild(file);
  file->name.setValue(fileName.c_str());
  // connect object OMBVMATERIAL in file to hdf5 mat if it is of type SoMaterial
  SoBase *ref=SoNode::getByName("BASECOLOR");
  if(ref && ref->getTypeId()==SoMaterial::getClassTypeId()) {
    ((SoMaterial*)ref)->diffuseColor.connectFrom(&mat->diffuseColor);
    ((SoMaterial*)ref)->specularColor.connectFrom(&mat->specularColor);
    ((SoMaterial*)ref)->shininess.connectFrom(&mat->shininess);
  }

  // scale ref/localFrame
  SoGetBoundingBoxAction bboxAction(SbViewportRegion(0,0));
  bboxAction.apply(soSep);
  float x1,y1,z1,x2,y2,z2;
  bboxAction.getBoundingBox().getBounds(x1,y1,z1,x2,y2,z2);
  double size=min(x2-x1,min(y2-y1,z2-z1));
  refFrameScale->scaleFactor.setValue(size,size,size);
  localFrameScale->scaleFactor.setValue(size,size,size);

  // outline
}
