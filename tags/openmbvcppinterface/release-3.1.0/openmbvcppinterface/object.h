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

#include <string>
#include <map>
#include <H5Cpp.h>
#include "openmbvcppinterfacetinyxml/tinyxml-src/tinyxml.h"
#include "openmbvcppinterface/doubleparam.h"

#define OPENMBVNS_ "http://openmbv.berlios.de/OpenMBV"
#define OPENMBVNS "{"OPENMBVNS_"}"

namespace OpenMBV {

  class Group;

  /** Abstract base class */
  class Object {
    friend class Group;
    protected:
      std::string name;
      std::string enableStr;
      Group* parent;
      virtual void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="")=0;
      virtual void createHDF5File()=0;
      H5::Group *hdf5Group;
      virtual void terminate()=0;
      static std::map<std::string, double> simpleParameter;
    public:
      /** Default constructor */
      Object();

      /** Virtual destructor */
      virtual ~Object();

      /** Enable this object in the viewer if true (the default) */
      void setEnable(bool enable) { enableStr=(enable==true)?"true":"false"; }

      /** Set the name of this object */
      void setName(const std::string& name_) { name=name_; }

      /** Returns the full name (path) of the object */
      std::string getFullName();

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(TiXmlElement *element);

      /** Clear the all parameters */
      static void clearSimpleParameters();

      /** Add a parameter */
      static void addSimpleParameter(std::string name, double value);
  };

}

#endif
