#ifndef _VIBODYCUBE_H_
#define _VIBODYCUBE_H_

#include "config.h"
#include "rigidbody.h"
#include <string>
#include "tinyxml.h"
#include <H5Cpp.h>

class IvBody : public RigidBody {
  Q_OBJECT
  public:
    IvBody(TiXmlElement* element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent);
};

#endif
