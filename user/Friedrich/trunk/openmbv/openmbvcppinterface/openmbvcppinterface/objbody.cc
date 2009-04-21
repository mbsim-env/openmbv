#include <openmbvcppinterface/objbody.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

ObjBody::ObjBody() : RigidBody(),
  useTextureFromMatLib(true),
  useMaterialFromMatLib(true),
  normals(fromObjFile),
  epsVertex(-1),
  epsNormal(-1),
  smoothBarrier(-1),
  outline(none) {
}

void ObjBody::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<ObjBody name=\""<<name<<"\">"<<endl;
    RigidBody::writeXMLFile(xmlFile, indent+"  ");
    xmlFile<<indent<<"  <objFileName>"<<objFileName<<"</objFileName>"<<endl;
    xmlFile<<indent<<"  <useTextureFromMatLib>"<<(useTextureFromMatLib?"true":"false")<<"</useTextureFromMatLib>"<<endl;
    xmlFile<<indent<<"  <useMaterialFromMatLib>"<<(useMaterialFromMatLib?"true":"false")<<"</useMaterialFromMatLib>"<<endl;
    string normalsStr;
    switch(normals) {
      case fromObjFile: normalsStr="fromObjFile"; break;
      case flat: normalsStr="flat"; break;
      case smooth: normalsStr="smooth"; break;
      case smoothIfLessBarrier: normalsStr="smoothIfLessBarrier"; break;
    }
    xmlFile<<indent<<"  <normals>"<<normalsStr<<"</normals>"<<endl;
    xmlFile<<indent<<"  <epsVertex>"<<epsVertex<<"</epsVertex>"<<endl;
    xmlFile<<indent<<"  <epsNormal>"<<epsNormal<<"</epsNormal>"<<endl;
    xmlFile<<indent<<"  <smoothBarrier>"<<smoothBarrier<<"</smoothBarrier>"<<endl;
    string outlineStr;
    switch(outline) {
      case none: outlineStr="none"; break;
      case calculate: outlineStr="calculate"; break;
      case fromFile: outlineStr="fromFile"; break;
    }
    xmlFile<<indent<<"  <outline>"<<outlineStr<<"</outline>"<<endl;
  xmlFile<<indent<<"</ObjBody>"<<endl;
}