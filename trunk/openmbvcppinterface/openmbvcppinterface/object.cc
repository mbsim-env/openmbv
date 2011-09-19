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

#include <openmbvcppinterface/object.h>
#include <openmbvcppinterface/group.h>
#include "openmbvcppinterfacetinyxml/tinyxml-src/tinynamespace.h"
#include "openmbvcppinterface/simpleparameter.h"
#include <assert.h>
#include <cmath>

using namespace std;
using namespace OpenMBV;

Object::Object() : name("NOTSET"), enableStr("true"), boundingBoxStr("false"), parent(0), hdf5Group(0) {
}

Object::~Object() {
  if(hdf5Group!=0) { delete hdf5Group; hdf5Group=0; }
}

string Object::getFullName() {
  if(parent)
    return parent->getFullName()+"/"+name;
  else
    return name;
}

void Object::initializeUsingXML(TiXmlElement *element) {
  setName(element->Attribute("name"));
  if(element->Attribute("enable") && 
     (element->Attribute("enable")==string("false") || element->Attribute("enable")==string("0")))
    setEnable(false);
  if(element->Attribute("boundingBox") && 
     (element->Attribute("boundingBox")==string("false") || element->Attribute("boundingBox")==string("0")))
    setBoundingBox(false);
}

// convenience: convert e.g. "[3;7;7.9]" to std::vector<double>(3,7,7.9)
vector<double> Object::toVector(string str) {
  for(unsigned int i=0; i<str.length(); i++)
    if(str[i]=='[' || str[i]==']' || str[i]==';') str[i]=' ';
  stringstream stream(str);
  double d;
  vector<double> ret;
  while(1) {
    stream>>d;
    if(stream.fail()) break;
    ret.push_back(d);
  }
  return ret;
}

// convenience: convert e.g. "[3,7;9,7.9]" to std::vector<std::vector<double> >
vector<vector<double> > Object::toMatrix(string str) {
  vector<vector<double> > ret;
  for(unsigned int i=0; i<str.length(); i++)
    if(str[i]=='[' || str[i]==']' || str[i]==',') str[i]=' ';
  bool br=false;
  while(1) {
    int end=str.find(';'); if(end<0) { end=str.length(); br=true; }
    ret.push_back(toVector(str.substr(0,end)));
    if(br) break;
    str=str.substr(end+1);
  }
  return ret;
}

ScalarParameter Object::getDouble(TiXmlElement *e) {
  // value is a parameter name
  if((string(e->GetText())[0]>='a' && string(e->GetText())[0]<='z') ||
     (string(e->GetText())[0]>='A' && string(e->GetText())[0]<='Z') ||
     (string(e->GetText())[0]>='_'))
    return ScalarParameter(e->GetText());
  // value is numeric
  vector<vector<double> > m=toMatrix(e->GetText());
  if(m.size()==1 && m[0].size()==1)
    return m[0][0];
  else {
    ostringstream str;
    str<<": Obtained matrix of size "<<m.size()<<"x"<<m[0].size()<<" ("<<e->GetText()<<") "<<
         "where a scalar was requested for element "<<e->ValueStr();
    TiXml_location(e, "", str.str());
    throw 1;
  }
  return NAN;
}

VectorParameter Object::getVec(TiXmlElement *e, unsigned int rows) {
  // value is a parameter name
  if((string(e->GetText())[0]>='a' && string(e->GetText())[0]<='z') ||
     (string(e->GetText())[0]>='A' && string(e->GetText())[0]<='Z') ||
     (string(e->GetText())[0]>='_'))
    return VectorParameter(e->GetText());
  // value is numeric
  vector<vector<double> > m=toMatrix(e->GetText());
  if((rows==0 || m.size()==rows) && m[0].size()==1) {
    vector<double> v;
    for(unsigned int i=0; i<m.size(); i++)
      v.push_back(m[i][0]);
    return v;
  }
  else {
    ostringstream str;
    str<<": Obtained matrix of size "<<m.size()<<"x"<<m[0].size()<<" ("<<e->GetText()<<") "<<
         "where a vector of size "<<rows<<" was requested for element "<<e->ValueStr();
    TiXml_location(e, "", str.str());
    throw 1;
  }
  return vector<double>();
}

MatrixParameter Object::getMat(TiXmlElement *e, unsigned int rows, unsigned int cols) {
  // value is a parameter name
  if((string(e->GetText())[0]>='a' && string(e->GetText())[0]<='z') ||
     (string(e->GetText())[0]>='A' && string(e->GetText())[0]<='Z') ||
     (string(e->GetText())[0]>='_'))
    return MatrixParameter(e->GetText());
  // value is numeric
  vector<vector<double> > m=toMatrix(e->GetText());
  if((rows==0 || m.size()==rows) && (cols==0 || m[0].size()==cols))
    return m;
  else {
    ostringstream str;
    str<<": Obtained matrix of size "<<m.size()<<"x"<<m[0].size()<<" ("<<e->GetText()<<") "<<
         "where a matrix of size "<<rows<<"x"<<cols<<" was requested for element "<<e->ValueStr();
    TiXml_location(e, "", str.str());
    throw 1;
  }
  return vector<vector<double> >();
}

TiXmlElement *Object::writeXMLFile(TiXmlNode *parent) {
  TiXmlElement *e=new TiXmlElement(getClassName());
  parent->LinkEndChild(e);
  addAttribute(e, "name", name);
  addAttribute(e, "enable", enableStr, string("true"));
  addAttribute(e, "boundingBox", boundingBoxStr, string("false"));
  return e;
}

Group* Object::getSeparateGroup() {
  return parent->getSeparateGroup();
}

Group* Object::getTopLevelGroup() {
  return parent->getTopLevelGroup();
}

double Object::get(const ScalarParameter& src) {
  if(src.getParamStr()=="")
    return src.getValue();
  else if(src.getParamStr()=="nan" || src.getParamStr()=="NaN" || src.getParamStr()=="NAN")
    return NAN;
  else
    return getSeparateGroup()->getScalarParameter(src.getParamStr());
}

vector<double> Object::get(const VectorParameter& src) {
  if(src.getParamStr()=="")
    return src.getValue();
  else if(src.getParamStr()=="nan" || src.getParamStr()=="NaN" || src.getParamStr()=="NAN")
    return vector<double>(1, NAN);
  else
    return getSeparateGroup()->getVectorParameter(src.getParamStr());
}

vector<vector<double> > Object::get(const MatrixParameter& src) {
  if(src.getParamStr()=="")
    return src.getValue();
  else if(src.getParamStr()=="nan" || src.getParamStr()=="NaN" || src.getParamStr()=="NAN")
    return vector<vector<double> >(1, vector<double>(1, NAN));
  else
    return getSeparateGroup()->getMatrixParameter(src.getParamStr());
}

void Object::set(ScalarParameter& dst, const ScalarParameter& src) {
  dst=src;
  if(dst.getParamStr()!="" && dst.getAddParameter()==true) scalarParameter[dst.getParamStr()]=dst.getValue();
}

void Object::set(VectorParameter& dst, const VectorParameter& src) {
  dst=src;
  if(dst.getParamStr()!="" && dst.getAddParameter()==true) vectorParameter[dst.getParamStr()]=dst.getValue();
}

void Object::set(MatrixParameter& dst, const MatrixParameter& src) {
  dst=src;
  if(dst.getParamStr()!="" && dst.getAddParameter()==true) matrixParameter[dst.getParamStr()]=dst.getValue();
}

void Object::collectParameter(map<string, double>& sp, map<string, vector<double> >& vp, map<string, vector<vector<double> > >& mp, bool collectAlsoSeparateGroup) {
  for(map<string, double>::iterator i=scalarParameter.begin(); i!=scalarParameter.end(); i++)
    sp[i->first]=i->second;
  for(map<string, vector<double> >::iterator i=vectorParameter.begin(); i!=vectorParameter.end(); i++)
    vp[i->first]=i->second;
  for(map<string, vector<vector<double> > >::iterator i=matrixParameter.begin(); i!=matrixParameter.end(); i++)
    mp[i->first]=i->second;
}

void Object::addElementText(TiXmlElement *parent, std::string name, double value, double def) {
  if(!(value==def || (isnan(def) && isnan(value))))
    addElementText(parent, name, value);
}

void Object::addElementText(TiXmlElement *parent, std::string name, SimpleParameter<double> value, double def) {
  if(!(get(value)==def || (isnan(def) && isnan(get(value)))) || value.getParamStr()!="")
    addElementText(parent, name, value);
}
