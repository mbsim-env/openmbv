#include <openmbvcppinterface/compoundrigidbody.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

CompoundRigidBody::CompoundRigidBody() : RigidBody() {
}

void CompoundRigidBody::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<CompoundRigidBody name=\""<<name<<"\" expand=\""<<expandStr<<"\">"<<endl;
    RigidBody::writeXMLFile(xmlFile, indent+"  ");
    for(int i=0; i<rigidBody.size(); i++)
      rigidBody[i]->writeXMLFile(xmlFile, indent+"  ");
  xmlFile<<indent<<"</CompoundRigidBody>"<<endl;
}
