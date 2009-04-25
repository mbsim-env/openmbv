#ifndef _CUBOID_H_
#define _CUBOID_H_

#include "config.h"
#include "rigidbody.h"
#include <string>
#include "tinyxml.h"
#include <H5Cpp.h>

class Cuboid : public RigidBody {
  Q_OBJECT
  public:
    Cuboid(TiXmlElement* element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent);
};

#endif
