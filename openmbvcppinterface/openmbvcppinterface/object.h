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

#ifndef _OPENMBV_OBJECT_H_
#define _OPENMBV_OBJECT_H_

#include <fmatvec/atom.h>
#include <string>
#include <sstream>
#include <openmbvcppinterface/objectfactory.h>
#include <hdf5serie/group.h>
#include <mbxmlutilshelper/dom.h>
#include <utility>
#include <vector>

namespace OpenMBV {

  typedef int Index;

#ifndef SWIG // SWIG can not parse this (swig bug?). However it is not needed for the swig interface -> removed for swig
  const MBXMLUtils::NamespaceURI OPENMBV("http://www.mbsim-env.de/OpenMBV");
#endif

  class Group;

  /** Abstract base class */
  class Object
#ifndef SWIG
    // with swig we do not need any access to fmatvec::Atom members. To avoid problems with other projects
    // deriving from fmatvec::Atomm we skip this completely from swig processing.
    : virtual public fmatvec::Atom
#endif
  {
    friend class Group;
    protected:
      std::string name;
      std::string enableStr, boundingBoxStr;
      std::string ID; // Note: the ID is metadata and stored as a processing instruction in XML
      std::string environmentStr;
      bool selected{false}; // Note: the selected flag is metadata and not stored in XML but used by OpenMBVGUI
      std::weak_ptr<Group> parent;

      virtual void createHDF5File()=0;
      virtual void openHDF5File()=0;
      H5::GroupBase *hdf5Group{nullptr};
      virtual void terminate()=0;

      Object();
      ~Object() override;
    public:
      /** Enable this object in the viewer if true (the default) */
      void setEnable(bool enable) { enableStr=(enable)?"true":"false"; }

      bool getEnable() { return enableStr=="true"?true:false; }

      /** Draw bounding box of this object in the viewer if true (the default) */
      void setBoundingBox(bool bbox) { boundingBoxStr=(bbox)?"true":"false"; }

      bool getBoundingBox() { return boundingBoxStr=="true"?true:false; }

      /** Set the name of this object */
      void setName(const std::string& name_) { name=name_; }

      std::string getName() { return name; }

      /** Returns the full name (path) of the object */
      virtual std::string getFullName(bool includingFileName=false, bool stopAtSeparateFile=false);

      /** If set to true than this object is an environment object:
       * a static object which has no time dependent part (does not read anything from the h5 file). */
      void setEnvironment(bool env) { environmentStr=(env)?"true":"false"; }

      bool getEnvironment() { return environmentStr=="true"?true:false; }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(xercesc::DOMElement *element);

      virtual xercesc::DOMElement *writeXMLFile(xercesc::DOMNode *parent);

      /** return the first Group in the tree which is an separateFile */
      std::shared_ptr<Group> getSeparateGroup();

      /** return the top level Group */
      std::shared_ptr<Group> getTopLevelGroup();

      std::weak_ptr<Group> getParent() { return parent; }

      H5::GroupBase *getHDF5Group() { return hdf5Group; };

      /** get the ID sting of the Object (Note: the ID is metadata and stored as a processing instruction in XML) */
      std::string getID() const { return ID; }
      /** set the ID sting of the Object (Note: the ID is metadata and stored as a processing instruction in XML) */
      void setID(std::string ID_) { ID=std::move(ID_); }

      /** get the selected flag (Note: the selected flag is metadata and not stored in XML but used by OpenMBVGUI) */
      bool getSelected() const { return selected; }
      /** set the selected flag (Note: the selected flag is metadata and not stored in XML but used by OpenMBVGUI) */
      void setSelected(bool selected_) { selected=selected_; }
  };

}

#endif
