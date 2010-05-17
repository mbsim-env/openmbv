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

#ifndef _OPENMBV_GROUP_H_
#define _OPENMBV_GROUP_H_

#include <openmbvcppinterface/object.h>
#include <vector>
#include <H5Cpp.h>

namespace OpenMBV {

  /** A container for bodies */
  class Group : public Object {
    friend class Body;
    protected:
      std::vector<Object*> object;
      std::string expandStr;
      bool separateFile;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
    public:
      /** Default constructor */
      Group();

      /** Expand this tree node in a view if true (the default) */
      void setExpand(bool expand) { expandStr=(expand==true)?"true":"false"; }

      /** Add a object to this object container */
      void addObject(Object* object);
      
      /** Plot a separate xml and h5 file for this group if truee */
      void setSeparateFile(bool sepFile) { separateFile=sepFile; }
      
      /** Initialisze/Write the XML file.
       * Call this function for the root node of the tree to create the XML file.
       */
      void initializeXML();
      
      /** Initialisze the h5 file.
       * Call this function for the root node of the tree to init the h5 file.
       */
      void initializeH5();
      
      /** Initialisze the tree (XML and h5).
       * Call this function for the root node of the tree before starting writing.
       */
      void initialize() { initializeXML(); initializeH5(); }

      /** terminate the tree.
       * Call this function for the root node of the free after all writing has done.
       */
      void terminate();
  };

}

#endif
