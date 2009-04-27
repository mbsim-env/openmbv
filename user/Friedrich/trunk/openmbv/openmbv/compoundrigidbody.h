#ifndef _COMPOUNDRIGIDBODY_H_
#define _COMPOUNDRIGIDBODY_H_

#include "config.h"
#include "rigidbody.h"
#include "tinyxml.h"
#include <H5Cpp.h>

class CompoundRigidBody : public RigidBody {
  Q_OBJECT
  public:
    CompoundRigidBody(TiXmlElement* element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent);
};

#endif
