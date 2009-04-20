#include <openmbvcppinterface/object.h>
#include <openmbvcppinterface/group.h>
#include <assert.h>

using namespace std;
using namespace OpenMBV;

Object::Object() : name(""), parent(0), hdf5Group(0) {
}

Object::~Object() {
  if(hdf5Group) delete hdf5Group;
}

string Object::getFullName() {
  if(parent)
    return parent->getFullName()+"/"+name;
  else
    return name;
}
