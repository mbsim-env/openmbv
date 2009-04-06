#include "objectfactory.h"
#include "iostream"
#include "group.h"
#include "cuboid.h"
#include "frame.h"
#include <string>

using namespace std;

Object *ObjectFactory(TiXmlElement *element) {
  if(element->ValueStr()==MBVISNS"Group")
    return new Group(element);
  else if(element->ValueStr()==MBVISNS"Cuboid")
    return new Cuboid(element);
  else if(element->ValueStr()==MBVISNS"Frame")
    return new Frame(element);
  cout<<"ERROR: Unknown element: "<<element->ValueStr()<<endl;
  assert(0);
}
