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

#include "config.h"
#include "openmbvcppinterface/object.h"
#include "openmbvcppinterface/group.h"
#include <xercesc/dom/DOMProcessingInstruction.hpp>
#include <assert.h>
#include <cmath>
#include <limits>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

// we use none signaling (quiet) NaN values for double in OpenMBVC++Interface -> Throw compile error if these do not exist.
BOOST_STATIC_ASSERT_MSG(numeric_limits<double>::has_quiet_NaN, "This platform does not support quiet NaN for double.");

Object::Object() : name("NOTSET"), enableStr("true"), boundingBoxStr("false"), ID(""), selected(false), hdf5Group(0) {
}

Object::~Object() {
}

string Object::getFullName(bool includingFileName, bool stopAtSeparateFile) {
  boost::shared_ptr<Group> p=parent.lock();
  if(p)
    return p->getFullName(includingFileName, stopAtSeparateFile)+"/"+name;
  else
    return name;
}

void Object::initializeUsingXML(DOMElement *element) {
  setName(E(element)->getAttribute("name"));
  if(E(element)->hasAttribute("enable") && 
     (E(element)->getAttribute("enable")=="false" || E(element)->getAttribute("enable")=="0"))
    setEnable(false);
  if(E(element)->hasAttribute("boundingBox") && 
     (E(element)->getAttribute("boundingBox")=="false" || E(element)->getAttribute("boundingBox")=="0"))
    setBoundingBox(false);

  DOMProcessingInstruction *ID = E(element)->getFirstProcessingInstructionChildNamed("OPENMBV_ID");
  if(ID)
    setID(X()%ID->getData());
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

double Object::getDouble(DOMElement *e) {
  string name = X()%E(e)->getFirstTextChild()->getData();
  vector<vector<double> > m=toMatrix(name);
  if(m.size()==1 && m[0].size()==1)
    return m[0][0];
  else {
    ostringstream str;
    str<<": Obtained matrix of size "<<m.size()<<"x"<<m[0].size()<<" ("<<name<<") "<<
         "where a scalar was requested for element "<<X()%e->getTagName();
    throw MBXMLUtils::DOMEvalException(str.str(), e);
  }
}

vector<double> Object::getVec(DOMElement *e, unsigned int rows) {
  string name = X()%E(e)->getFirstTextChild()->getData();
  vector<vector<double> > m=toMatrix(name);
  if((rows==0 || m.size()==rows) && m[0].size()==1) {
    vector<double> v;
    for(unsigned int i=0; i<m.size(); i++)
      v.push_back(m[i][0]);
    return v;
  }
  else {
    ostringstream str;
    str<<": Obtained matrix of size "<<m.size()<<"x"<<m[0].size()<<" ("<<name<<") "<<
         "where a vector of size "<<rows<<" was requested for element "<<X()%e->getTagName();
    throw MBXMLUtils::DOMEvalException(str.str(), e);
  }
  return vector<double>();
}

std::vector<std::vector<double> > Object::getMat(DOMElement *e, unsigned int rows, unsigned int cols) {
  string name = X()%E(e)->getFirstTextChild()->getData();
  vector<vector<double> > m=toMatrix(name);
  if((rows==0 || m.size()==rows) && (cols==0 || m[0].size()==cols))
    return m;
  else {
    ostringstream str;
    str<<": Obtained matrix of size "<<m.size()<<"x"<<m[0].size()<<" ("<<name<<") "<<
         "where a matrix of size "<<rows<<"x"<<cols<<" was requested for element "<<X()%e->getTagName();
    throw MBXMLUtils::DOMEvalException(str.str(), e);
  }
  return vector<vector<double> >();
}

DOMElement *Object::writeXMLFile(DOMNode *parent) {
  DOMDocument *doc=parent->getNodeType()==DOMNode::DOCUMENT_NODE ? static_cast<DOMDocument*>(parent) : parent->getOwnerDocument();
  DOMElement *e=D(doc)->createElement(OPENMBV%getClassName());
  parent->insertBefore(e, NULL);
  addAttribute(e, "name", name);
  addAttribute(e, "enable", enableStr, string("true"));
  addAttribute(e, "boundingBox", boundingBoxStr, string("false"));
  if(!ID.empty()) {
    DOMDocument *doc=parent->getOwnerDocument();
    DOMProcessingInstruction *id=doc->createProcessingInstruction(X()%"OPENMBV_ID", X()%ID);
    e->insertBefore(id, NULL);
  }
  return e;
}

boost::shared_ptr<Group> Object::getSeparateGroup() {
  return parent.lock()->getSeparateGroup();
}

boost::shared_ptr<Group> Object::getTopLevelGroup() {
  return parent.lock()->getTopLevelGroup();
}

void Object::addElementText(DOMElement *parent, const MBXMLUtils::FQN &name, double value, double def) {
  if(!(value==def || (isnan(def) && isnan(value))))
    addElementText(parent, name, value);
}

void Object::addElementText(DOMElement *parent, const MBXMLUtils::FQN &name, const vector<double> &value) {
  std::ostringstream oss;
  for(vector<double>::const_iterator ele=value.begin(); ele!=value.end(); ++ele)
    oss<<(ele==value.begin()?"[":"; ")<< *ele;
  oss<<"]";
  xercesc::DOMElement *ele = MBXMLUtils::D(parent->getOwnerDocument())->createElement(name);
  ele->insertBefore(parent->getOwnerDocument()->createTextNode(MBXMLUtils::X()%oss.str()), NULL);
  parent->insertBefore(ele, NULL);
}

void Object::addElementText(DOMElement *parent, const MBXMLUtils::FQN &name, const vector<vector<double> > &value) {
  std::ostringstream oss;
  for(vector<vector<double> >::const_iterator row=value.begin(); row!=value.end(); ++row)
    for(vector<double>::const_iterator ele=row->begin(); ele!=row->end(); ++ele)
      oss<<(row==value.begin() && ele==row->begin()?"[":(ele==row->begin()?"; ":", "))<< *ele;
  oss<<"]";
  xercesc::DOMElement *ele = MBXMLUtils::D(parent->getOwnerDocument())->createElement(name);
  ele->insertBefore(parent->getOwnerDocument()->createTextNode(MBXMLUtils::X()%oss.str()), NULL);
  parent->insertBefore(ele, NULL);
}

}
