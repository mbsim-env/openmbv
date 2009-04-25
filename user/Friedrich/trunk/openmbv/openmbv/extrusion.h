#ifndef _EXTRUSION_H_
#define _EXTRUSION_H_

#include "config.h"
#include "rigidbody.h"
#include <string>
#include "tinyxml.h"
#include <H5Cpp.h>

class Extrusion : public RigidBody {
  Q_OBJECT
  public:
    Extrusion(TiXmlElement* element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent);
};

#endif
