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

#include <openmbvcppinterface/body.h>
#include <iostream>
#include <fstream>
#include <H5Cpp.h>
#include <openmbvcppinterface/group.h>
#include "openmbvcppinterfacetinyxml/tinyxml-src/tinynamespace.h"
#include <cmath>

using namespace std;
using namespace OpenMBV;

Body::Body() : Object(),
  hdf5LinkBody(0) {
}

void Body::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  if(hdf5LinkBody)
    xmlFile<<indent<<"<hdf5Link ref=\""<<getRelPathTo(hdf5LinkBody)<<"\"/>"<<endl;
}

void Body::createHDF5File() {
  if(!hdf5LinkBody)
    hdf5Group=new H5::Group(parent->hdf5Group->createGroup(name));
  else
    parent->hdf5Group->link(H5L_TYPE_SOFT, getRelPathTo(hdf5LinkBody), name);
}

std::string Body::getRelPathTo(Body* destBody) {
  // create relative path to destination
  string dest=destBody->getFullName();
  string src=getFullName();
  string reldest="";
  while(dest.substr(0,dest.find('/',1))==src.substr(0,src.find('/',1)))  {
    dest=dest.substr(dest.find('/',1));
    src=src.substr(src.find('/',1));
  }
  while((signed)src.find('/',1)>=0) {
    reldest=reldest+"../";
    src=src.substr(src.find('/',1));
  }
  reldest=reldest+dest.substr(1);
  return reldest;
}

void Body::terminate() {
}

// convenience: convert e.g. "[3;7;7.9]" to std::vector<double>(3,7,7.9)
vector<double> Body::toVector(string str) {
  for(int i=0; i<str.length(); i++)
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
vector<vector<double> > Body::toMatrix(string str) {
  vector<vector<double> > ret;
  for(int i=0; i<str.length(); i++)
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

double Body::getDouble(TiXmlElement *e) {
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

vector<double> Body::getVec(TiXmlElement *e, int rows) {
  vector<vector<double> > m=toMatrix(e->GetText());
  if((rows==0 || m.size()==rows) && m[0].size()==1) {
    vector<double> v;
    for(int i=0; i<m.size(); i++)
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

vector<vector<double> > Body::getMat(TiXmlElement *e, int rows, int cols) {
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

void Body::initializeUsingXML(TiXmlElement *element) {
  Object::initializeUsingXML(element);
}
