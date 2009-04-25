#include <openmbvcppinterface/frustum.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

Frustum::Frustum() : RigidBody(),
  baseRadius(1),
  topRadius(1),
  height(2),
  innerBaseRadius(0),
  innerTopRadius(0) {
}

void Frustum::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<Frustum name=\""<<name<<"\" name=\""<<expandStr<<"\">"<<endl;
    RigidBody::writeXMLFile(xmlFile, indent+"  ");
    xmlFile<<indent<<"  <baseRadius>"<<baseRadius<<"</baseRadius>"<<endl;
    xmlFile<<indent<<"  <topRadius>"<<topRadius<<"</topRadius>"<<endl;
    xmlFile<<indent<<"  <height>"<<height<<"</height>"<<endl;
    xmlFile<<indent<<"  <innerBaseRadius>"<<innerBaseRadius<<"</innerBaseRadius>"<<endl;
    xmlFile<<indent<<"  <innerTopRadius>"<<innerTopRadius<<"</innerTopRadius>"<<endl;
  xmlFile<<indent<<"</Frustum>"<<endl;
}
