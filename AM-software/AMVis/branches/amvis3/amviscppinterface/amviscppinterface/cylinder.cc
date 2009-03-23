#include <amviscppinterface/cylinder.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace AMVis;

Cylinder::Cylinder() : RigidBody(),
  baseRadius(1),
  topRadius(1),
  height(2),
  innerBaseRadius(-1),
  innerTopRadius(-1) {
}

void Cylinder::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<Cylinder name=\""<<name<<"\">"<<endl;
    RigidBody::writeXMLFile(xmlFile, indent+"  ");
    xmlFile<<indent<<"  <baseRadius>"<<baseRadius<<"</baseRadius>"<<endl;
    xmlFile<<indent<<"  <topRadius>"<<topRadius<<"</topRadius>"<<endl;
    xmlFile<<indent<<"  <height>"<<height<<"</height>"<<endl;
    xmlFile<<indent<<"  <innerBaseRadius>"<<innerBaseRadius<<"</innerBaseRadius>"<<endl;
    xmlFile<<indent<<"  <innerTopRadius>"<<innerTopRadius<<"</innerTopRadius>"<<endl;
  xmlFile<<indent<<"</Cylinder>"<<endl;
}
