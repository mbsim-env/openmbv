#ifndef _OBJBODY_H_
#define _OBJBODY_H_

#include "config.h"
#include "rigidbody.h"
#include <string>
#include "tinyxml.h"
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoNormal.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/nodes/SoIndexedLineSet.h>
#include <Inventor/nodes/SoTexture2.h>
#include <Inventor/nodes/SoTextureCoordinate2.h>
#include <H5Cpp.h>

class ObjBody : public RigidBody {
  Q_OBJECT
  protected:
    enum Normals {
      fromObjFile,
      flat,
      smooth,
      smoothIfLessBarrier
    };
    enum Outline {
      none,
      calculate,
      fromFile
    };
    SoCoordinate3 *v;
    SoTextureCoordinate2 *t;
    SoNormal *n;
    class MtlMapGroup {
      public:
        MtlMapGroup();
        SoIndexedFaceSet *f;
        SoIndexedLineSet *ol;
        SoNormal *n;
        SoMaterial *mat;
        SoTexture2 *map;
    };
    double smoothBarrier;
    void readMtlLib(const std::string& mtlFile, std::map<QString, SoMaterial*>& material);
    void readMapLib(const std::string& mtlFile, std::map<QString, SoTexture2*>& map_);
  public:
    ObjBody(TiXmlElement* element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent);
};

#endif
