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
#include <map>

namespace OpenMBV {

  /** A container for bodies */
  class Group : public Object
#ifndef SWIG
      , public std::enable_shared_from_this<Group>
#endif
    {
    friend class Body;
    friend class Object;
    friend class ObjectFactory;
    protected:
      std::vector<std::shared_ptr<Object> > object;
      std::string expandStr;
      std::string fileName; // the file name of the .ombv.xml file of this separateFile Group including the absolute or relatvie path
      bool separateFile;
      std::shared_ptr<H5::File> hdf5File;
      void createHDF5File();
      void openHDF5File();

      Group();
      virtual ~Group();

      /** Initialisze/Write the XML file.
       * Call this function for the root node of the tree to create/write/ the XML file.
       */
      void writeXML();

      /** Read the XML file.
       * Call this function to read an OpenMBV XML file and creating the Object tree.
       */
      void readXML();
      
      /** Initialisze/Write the h5 file.
       * Call this function for the root node of the tree to init the h5 file.
       * (Only the h5 tree i written, but do data. Use append() to write date to h5 file after calling this function)
       */
      void writeH5();

      /** Read/open an existing h5 file. Before calling this function readXML() must be called or simply call read().
       */
      void readH5();

    public:
      /** Expand this tree node in a view if true (the default) */
      void setExpand(bool expand) { expandStr=(expand==true)?"true":"false"; }

      bool getExpand() { return expandStr=="true"?true:false; }

      /** Add a object to this object container */
      void addObject(std::shared_ptr<Object> object);

      std::vector<std::shared_ptr<Object> >& getObjects() {
        return object;
      }
      
      /** Plot a separate xml and h5 file for this group if truee */
      void setSeparateFile(bool sepFile) { separateFile=sepFile; }

      bool getSeparateFile() { return separateFile; }
      std::shared_ptr<H5::File>& getHDF5File() { return hdf5File; }

      /** Returns the file name of the .ombv.xml file of this separateFile Group
       * including the absolute or relatvie path */
      std::string getFileName() { return fileName; }

      std::string getFullName(bool includingFileName=false, bool stopAtSeparateFile=false);
      
      /** Sets the file name of the .ombv.xml file of this separateFile Group
       * including the absolute or relatvie path */
      void setFileName(const std::string &fn) { fileName=fn; }
      
      /** Initialisze/Wrtie the tree (XML and h5).
       * This function simply calls writeXML() and writeH5().
       */
      void write(bool writeXMLFile=true, bool writeH5File=true);

      /** Read the tree (XML and h5).
       * This function simply calls readXML() and readH5().
       */
      void read(bool readXMLFile=true, bool readH5File=true);

      /** terminate the tree.
       * Call this function for the root node of the free after all writing has done.
       */
      void terminate();

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(xercesc::DOMElement *element);

      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent);

      /** return the first Group in the tree which is an separateFile */
      std::shared_ptr<Group> getSeparateGroup() {
        return separateFile?shared_from_this():parent.lock()->getSeparateGroup();
      }

      /** return the top level Group */
      std::shared_ptr<Group> getTopLevelGroup() {
        std::shared_ptr<Group> p=parent.lock();
        return !p?shared_from_this():p->getTopLevelGroup();
      }
  };

}

#endif
