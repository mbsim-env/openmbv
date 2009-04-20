#include <openmbvcppinterface/ivbody.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

IvBody::IvBody() : RigidBody() {
}

void IvBody::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<IvBody name=\""<<name<<"\">"<<endl;
    RigidBody::writeXMLFile(xmlFile, indent+"  ");
    xmlFile<<indent<<"  <ivFileName>"<<ivFileName<<"</ivFileName>"<<endl;
  xmlFile<<indent<<"</IvBody>"<<endl;
}
