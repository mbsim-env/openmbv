#include <openmbvcppinterface/sphere.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

Sphere::Sphere() : RigidBody(),
  radius(1) {
}

void Sphere::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<Sphere name=\""<<name<<"\" expand=\""<<expandStr<<"\">"<<endl;
    RigidBody::writeXMLFile(xmlFile, indent+"  ");
    xmlFile<<indent<<"  <radius>"<<radius<<"</radius>"<<endl;
  xmlFile<<indent<<"</Sphere>"<<endl;
}
