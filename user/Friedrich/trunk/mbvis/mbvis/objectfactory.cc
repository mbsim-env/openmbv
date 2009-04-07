#include "objectfactory.h"
#include "iostream"
#include "group.h"
#include "cuboid.h"
#include "frame.h"
#include <string>

using namespace std;

Object *ObjectFactory(TiXmlElement *element, H5::Group *h5Parent) {
  if(element->ValueStr()==MBVISNS"Group")
    return new Group(element, h5Parent);
  else if(element->ValueStr()==MBVISNS"Cuboid")
    return new Cuboid(element, h5Parent);
  else if(element->ValueStr()==MBVISNS"Frame")
    return new Frame(element, h5Parent);
  cout<<"ERROR: Unknown element: "<<element->ValueStr()<<endl;
  assert(0);
}
