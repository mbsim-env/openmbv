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
#include "group.h"
#include "objectfactory.h"
#include "mainwindow.h"
#include "compoundrigidbody.h"
#include <string>

using namespace std;

Group::Group(TiXmlElement* element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : Object(element, h5Parent, parentItem, soParent) {
  iconFile=":/group.svg";
  setIcon(0, QIconCached(iconFile.c_str()));

  if(dynamic_cast<CompoundRigidBody*>(parentItem)==0) {
    // expand or collapse
    if(element->Attribute("expand")==0)
      setExpanded(true);
    else
      if((element->Attribute("expand")==string("true") || element->Attribute("expand")==string("1")))
        setExpanded(true);
      else
        setExpanded(false);
  }

  // if xml:base attribute exist => new sub file
  if(element->Attribute("xml:base")) {
    iconFile=":/h5file.svg";
    setIcon(0, QIconCached(iconFile.c_str()));
    setText(0, element->Attribute("xml:base"));
  }
  // read XML
  TiXmlElement *e=element->FirstChildElement();
  while(e!=0) {
    if(e->ValueStr()==OPENMBVNS"Group" && e->FirstChildElement()==0) { e=e->NextSiblingElement(); continue; } // a hack for openmbvdeleterows.sh
    ObjectFactory(e, h5Group, this, soSep);
    e=e->NextSiblingElement();
  }

  // hide groups without childs
  if(childCount()==0) setHidden(true);
}

QString Group::getInfo() {
  return Object::getInfo()+
         QString("-----<br/>")+
         QString("<b>Number of Children:</b> %1<br/>").arg(childCount());
}
