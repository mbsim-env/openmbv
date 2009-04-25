#include "config.h"
#include "object.h"
#include "tinyxml.h"
#include <H5Cpp.h>

Object *ObjectFactory(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem* parentItem, SoGroup *soParent);
