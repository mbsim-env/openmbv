#ifndef _FRUSTUM_H_
#define _FRUSTUM_H_

#include "config.h"
#include "rigidbody.h"
#include <string>
#include "tinyxml.h"
#include <H5Cpp.h>

class Frustum : public RigidBody {
  Q_OBJECT
  public:
    Frustum(TiXmlElement* element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent);
};

#endif
