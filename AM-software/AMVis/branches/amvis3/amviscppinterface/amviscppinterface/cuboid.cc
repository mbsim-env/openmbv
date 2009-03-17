#include <amviscppinterface/cuboid.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace AMVis;

Cuboid::Cuboid(const string& name_) : RigidBody(name_),
  length(3, 1) {
}

void Cuboid::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<Cuboid name=\""<<name<<"\">"<<endl;
    RigidBody::writeXMLFile(xmlFile, indent+"  ");
    xmlFile<<indent<<"  <length>["<<length[0]<<";"
                               <<length[1]<<";"
                               <<length[2]<<"]</length>"<<endl;
  xmlFile<<indent<<"</Cuboid>"<<endl;
}
