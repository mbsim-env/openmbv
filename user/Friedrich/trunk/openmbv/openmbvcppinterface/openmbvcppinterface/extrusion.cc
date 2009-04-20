#include <openmbvcppinterface/extrusion.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

Extrusion::Extrusion() : RigidBody(),
  windingRule(odd),
  height(1) {
}

void Extrusion::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<Extrusion name=\""<<name<<"\">"<<endl;
    RigidBody::writeXMLFile(xmlFile, indent+"  ");
    string windingRuleStr;
    switch(windingRule) {
      case odd: windingRuleStr="odd"; break;
      case nonzero: windingRuleStr="nonzero"; break;
      case positive: windingRuleStr="positive"; break;
      case negative: windingRuleStr="negative"; break;
      case absGEqTwo: windingRuleStr="absGEqTwo"; break;
    }
    xmlFile<<indent<<"  <windingRule>"<<windingRuleStr<<"</windingRule>"<<endl;
    xmlFile<<indent<<"  <height>"<<height<<"</height>"<<endl;
    for(vector<vector<PolygonPoint*>*>::const_iterator i=contour.begin(); i!=contour.end(); i++) {
      xmlFile<<indent<<"  <contour>"<<endl;
      xmlFile<<indent<<"    [ ";
      for(vector<PolygonPoint*>::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
        if(j!=(*i)->begin()) xmlFile<<indent<<"      ";
        xmlFile<<(*j)->x<<", "<<(*j)->y<<", "<<(*j)->b;
        if(j+1!=(*i)->end()) xmlFile<<";"<<endl; else xmlFile<<" ]"<<endl;
      }
      xmlFile<<indent<<"  </contour>"<<endl;
    }
  xmlFile<<indent<<"</Extrusion>"<<endl;
}
