#include <openmbvcppinterface/rotation.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

Rotation::Rotation() : RigidBody(),
  contour(0) {
}

void Rotation::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<Rotation name=\""<<name<<"\">"<<endl;
    RigidBody::writeXMLFile(xmlFile, indent+"  ");
    if(contour) {
      xmlFile<<indent<<"  <contour>"<<endl;
      xmlFile<<indent<<"    [ ";
      for(vector<PolygonPoint*>::const_iterator j=contour->begin(); j!=contour->end(); j++) {
        if(j!=contour->begin()) xmlFile<<indent<<"      ";
        xmlFile<<(*j)->x<<", "<<(*j)->y<<", "<<(*j)->b;
        if(j+1!=contour->end()) xmlFile<<";"<<endl; else xmlFile<<" ]"<<endl;
      }
      xmlFile<<indent<<"  </contour>"<<endl;
    }
  xmlFile<<indent<<"</Rotation>"<<endl;
}
