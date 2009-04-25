#include <openmbvcppinterface/cube.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

Cube::Cube() : RigidBody(),
  length(1) {
}

void Cube::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<Cube name=\""<<name<<"\" name=\""<<expandStr<<"\">"<<endl;
    RigidBody::writeXMLFile(xmlFile, indent+"  ");
    xmlFile<<indent<<"  <length>"<<length<<"</length>"<<endl;
  xmlFile<<indent<<"</Cube>"<<endl;
}
