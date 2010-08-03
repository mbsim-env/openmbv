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

#include <openmbvcppinterface/group.h>
#include <openmbvcppinterface/body.h>
#include <openmbvcppinterface/objectfactory.h>
#include <hdf5serie/fileserie.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <H5Cpp.h>
#include "openmbvcppinterfacetinyxml/tinyxml-src/tinynamespace.h"

using namespace OpenMBV;
using namespace std;

Group::Group() : Object(), expandStr("true"), separateFile(false), topLevelFile(false) {
}

void Group::addObject(Object* newObject) {
  assert(newObject->name!="");
  for(unsigned int i=0; i<object.size(); i++)
    assert(object[i]->name!=newObject->name);
  object.push_back(newObject);
  newObject->parent=this;
}

TiXmlElement *Group::writeXMLFile(TiXmlNode *parent) {
  if(!separateFile) {
    TiXmlElement *e=Object::writeXMLFile(parent);
    addAttribute(e, "expand", expandStr, "true");
    for(unsigned int i=0; i<object.size(); i++)
      object[i]->writeXMLFile(e);
  }
  else {
    string fullName=getFullName();
    for(unsigned int i=0; i<fullName.length(); i++) if(fullName[i]=='/') fullName[i]='.';
    // create link (embed) in current xml file
    TiXmlElement *inc=new TiXmlElement("xi:include");
    parent->LinkEndChild(inc);
    inc->SetAttribute("href", fullName+".ombv.xml");
    // get directory of top level file
    string dir=getTopLevelGroup()->getFileName();
    size_t pos=dir.find_last_of('/');
    if(pos!=string::npos)
      dir=dir.substr(0, pos+1);
    else
      dir="";
    // create new xml file and write to it till now
    TiXmlDocument xmlFile(dir+fullName+".ombv.xml");
      xmlFile.LinkEndChild(new TiXmlDeclaration("1.0","UTF-8",""));
      TiXmlElement *e=Object::writeXMLFile(&xmlFile);
      addAttribute(e, "expand", expandStr, "true");
      e->SetAttribute("xmlns", OPENMBVNS_);
      e->SetAttribute("xmlns:xi", "http://www.w3.org/2001/XInclude");
      for(unsigned int i=0; i<object.size(); i++)
        object[i]->writeXMLFile(e);
    xmlFile.SaveFile();

    writeSimpleParameter(dir+fullName+".ombv.xml");
  }
  return 0;
}

void Group::createHDF5File() {
  if(!separateFile) {
    hdf5Group=new H5::Group(parent->hdf5Group->createGroup(name));
    for(unsigned int i=0; i<object.size(); i++)
      object[i]->createHDF5File();
  }
  else {
    string fullName=getFullName();
    for(unsigned int i=0; i<fullName.length(); i++) if(fullName[i]=='/') fullName[i]='.';
    // create link in current h5 file
    H5Lcreate_external((fullName+".ombv.h5").c_str(), "/",
                       parent->hdf5Group->getId(), name.c_str(),
                       H5P_DEFAULT, H5P_DEFAULT);
    // create new h5 file land write to in till now
    hdf5Group=(H5::Group*)new H5::FileSerie(fullName+".ombv.h5", H5F_ACC_TRUNC);
    for(unsigned int i=0; i<object.size(); i++)
      object[i]->createHDF5File();
  }
}

void Group::openHDF5File() {
  if(topLevelFile)
    hdf5Group=(H5::Group*)new H5::FileSerie(getFileName().substr(0,getFileName().length()-4)+".h5", H5F_ACC_RDONLY);
  else
    hdf5Group=new H5::Group(parent->hdf5Group->openGroup(name));
  for(unsigned int i=0; i<object.size(); i++)
    object[i]->openHDF5File();
}

void Group::writeXML() {
  if(fileName=="") fileName=name+".ombv.xml";
  separateFile=true;
  topLevelFile=true;
  // write .ombv.xml file
  setTopLevelFile(true);
  TiXmlDocument xmlFile(fileName);
    xmlFile.LinkEndChild(new TiXmlDeclaration("1.0","UTF-8",""));
    TiXmlElement *parent=Object::writeXMLFile(&xmlFile);
    addAttribute(parent, "expand", expandStr, "true");
    parent->SetAttribute("xmlns", OPENMBVNS_);
    parent->SetAttribute("xmlns:xi", "http://www.w3.org/2001/XInclude");
    for(unsigned int i=0; i<object.size(); i++)
      object[i]->writeXMLFile(parent);
  xmlFile.SaveFile();

  writeSimpleParameter(fileName);
}

void Group::writeH5() {
  hdf5Group=(H5::Group*)new H5::FileSerie(name+".ombv.h5", H5F_ACC_TRUNC);
  for(unsigned int i=0; i<object.size(); i++)
    object[i]->createHDF5File();
  hdf5Group->flush(H5F_SCOPE_GLOBAL);
}

void Group::terminate() {
  for(unsigned int i=0; i<object.size(); i++)
    object[i]->terminate();
  if(!separateFile)
    delete hdf5Group;
  else
    delete (H5::FileSerie*)hdf5Group;
  hdf5Group=0;
}

void Group::initializeUsingXML(TiXmlElement *element) {
  Object::initializeUsingXML(element);
  if(element->Attribute("expand") && 
     (element->Attribute("expand")==string("false") || element->Attribute("expand")==string("0")))
    setExpand(false);
  if(element->Attribute("xml:base")) {
    setSeparateFile(true);
    fileName=element->Attribute("xml:base");
    readSimpleParameter(fileName.substr(0,fileName.length()-4)+".param.xml");
  }

  TiXmlElement *e;
  e=element->FirstChildElement();
  while (e) {
    Object* obj=ObjectFactory::createObject(e);
    obj->initializeUsingXML(e);
    addObject(obj);
    e=e->NextSiblingElement();
  }
}

Group* Group::readXML(std::string fileName) {
  // read XML
  TiXmlDocument doc;
  doc.LoadFile(fileName); TiXml_PostLoadFile(&doc);
  map<string,string> dummy;
  incorporateNamespace(doc.FirstChildElement(), dummy);

  // read XML using OpenMBVCppInterface
  Group *rootGroup=new OpenMBV::Group;
  rootGroup->initializeUsingXML(doc.FirstChildElement());
  rootGroup->setTopLevelFile(true);
  return rootGroup;
}

void Group::readH5(Group *rootGrp) {
  rootGrp->openHDF5File();
}

void Group::readSimpleParameter(std::string filename) {
  TiXmlDocument *paramdoc=new TiXmlDocument;
  FILE *f=fopen(filename.c_str(),"r");
  if(f!=NULL) {
    fclose(f);
    paramdoc->LoadFile(filename); TiXml_PostLoadFile(paramdoc);
    TiXmlElement *e=paramdoc->FirstChildElement();
    map<string,string> dummy;
    incorporateNamespace(e,dummy);

    for(e=e->FirstChildElement(); e!=0; e=e->NextSiblingElement()) {
      if(e->ValueStr()==MBXMLUTILSPARAMNS"scalarParameter")
        scalarParameter[e->Attribute("name")]=atof(e->GetText());
      else if(e->ValueStr()==MBXMLUTILSPARAMNS"vectorParameter")
        vectorParameter[e->Attribute("name")]=toVector(e->GetText());
      else if(e->ValueStr()==MBXMLUTILSPARAMNS"matrixParameter")
        matrixParameter[e->Attribute("name")]=toMatrix(e->GetText());
    }
  }
}

double Group::getScalarParameter(std::string name) {
  return scalarParameter[name];
}

vector<double> Group::getVectorParameter(std::string name) {
  return vectorParameter[name];
}

vector<vector<double> > Group::getMatrixParameter(std::string name) {
  return matrixParameter[name];
}

void Group::writeSimpleParameter(std::string fileName) {
  collectParameter(scalarParameter, vectorParameter, matrixParameter, true);
  // write .ombv.param.xml file if simple parameters exist
  if(scalarParameter.size()>0 || vectorParameter.size()>0 || matrixParameter.size()>0) {
    TiXmlDocument xmlDoc(fileName.substr(0,fileName.length()-4)+".param.xml");
      xmlDoc.LinkEndChild(new TiXmlDeclaration("1.0","UTF-8",""));
      TiXmlElement *rootEle=new TiXmlElement("parameter");
      xmlDoc.LinkEndChild(rootEle);
      rootEle->SetAttribute("xmlns", MBXMLUTILSPARAMNS_);
      for(map<string,double>::iterator i=scalarParameter.begin(); i!=scalarParameter.end(); i++) {
        addElementText(rootEle, "scalarParameter", i->second);
        addAttribute(rootEle->LastChild(), "name", i->first);
      }
      for(map<string,vector<double> >::iterator i=vectorParameter.begin(); i!=vectorParameter.end(); i++) {
        addElementText(rootEle, "vectorParameter", i->second);
        addAttribute(rootEle->LastChild(), "name", i->first);
      }
      for(map<string,vector<vector<double> > >::iterator i=matrixParameter.begin(); i!=matrixParameter.end(); i++) {
        addElementText(rootEle, "matrixParameter", i->second);
        addAttribute(rootEle->LastChild(), "name", i->first);
      }
    xmlDoc.SaveFile();
  }
}

void Group::collectParameter(map<string, double>& sp, map<string, vector<double> >& vp, map<string, vector<vector<double> > >& mp, bool collectAlsoSeparateGroup) {
  if(!separateFile || collectAlsoSeparateGroup)
    for(size_t i=0; i<object.size(); i++)
      object[i]->collectParameter(sp, vp, mp);
}
