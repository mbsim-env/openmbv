#include <openmbvcppinterface/invisiblebody.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

InvisibleBody::InvisibleBody() : RigidBody() {
}

void InvisibleBody::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<InvisibleBody name=\""<<name<<"\" name=\""<<expandStr<<"\">"<<endl;
    RigidBody::writeXMLFile(xmlFile, indent+"  ");
  xmlFile<<indent<<"</InvisibleBody>"<<endl;
}
