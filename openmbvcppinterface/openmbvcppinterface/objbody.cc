/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

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

void ObjBody::initializeUsingXML(TiXmlElement *element) {
  RigidBody::initializeUsingXML(element);
  TiXmlElement *e;
  e=element->FirstChildElement(OPENMBVNS"objFileName");
  setObjFileName(e->GetText());
  e=element->FirstChildElement(OPENMBVNS"useTextureFromMatLib");
  setUseTextureFromMatLib((e->GetText()==string("true"))?true:false);
  e=element->FirstChildElement(OPENMBVNS"useMaterialFromMatLib");
  setUseMaterialFromMatLib((e->GetText()==string("true"))?true:false);
  e=element->FirstChildElement(OPENMBVNS"normals");
  string normalStr=e->GetText();
  if(normalStr=="fromObjFile") setNormals(fromObjFile);
  if(normalStr=="flat") setNormals(flat);
  if(normalStr=="smooth") setNormals(smooth);
  if(normalStr=="smoothIfLessBarrier") setNormals(smoothIfLessBarrier);
  e=element->FirstChildElement(OPENMBVNS"epsVertex");
  setEpsVertex(toVector(e->GetText())[0]);
  e=element->FirstChildElement(OPENMBVNS"epsNormal");
  setEpsNormal(toVector(e->GetText())[0]);
  e=element->FirstChildElement(OPENMBVNS"smoothBarrier");
  setSmoothBarrier(toVector(e->GetText())[0]);
  e=element->FirstChildElement(OPENMBVNS"outline");
  string outlineStr=e->GetText();
  if(outlineStr=="none") setOutline(none);
  if(outlineStr=="calculate") setOutline(calculate);
  if(outlineStr=="fromFile") setOutline(fromFile);
}
