#include "config.h"
#include "ivbody.h"
#include "tinynamespace.h"
#include <Inventor/nodes/SoFile.h>
#include <Inventor/nodes/SoDrawStyle.h>

#include <Inventor/fields/SoMFColor.h>
#include <Inventor/actions/SoWriteAction.h>

#include <vector>

using namespace std;

IvBody::IvBody(TiXmlElement *element, H5::Group *h5Parent) : RigidBody(element, h5Parent) {
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

  // outline
}
