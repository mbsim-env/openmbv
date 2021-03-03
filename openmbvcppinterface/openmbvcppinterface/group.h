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
      std::string fileName; // the file name of the .ombvx file of this separateFile Group including the absolute or relatvie path
      bool separateFile{false};
      std::shared_ptr<H5::File> hdf5File;
      std::function<void()> closeRequestCallback;
      void createHDF5File() override;
      void openHDF5File() override;

      Group();
      ~Group() override;

      /** Initialisze/Write the XML file.
       * Call this function for the root node of the tree to create/write/ the XML file.
       */
      void writeXML();

      /** Read the XML file.
       * Call this function to read an OpenMBV XML file and creating the Object tree.
       */
      void readXML();

    public:
      /** Expand this tree node in a view if true (the default) */
      void setExpand(bool expand) { expandStr=(expand)?"true":"false"; }

      bool getExpand() { return expandStr=="true"?true:false; }

      /** Add a object to this object container */
      void addObject(const std::shared_ptr<Object>& newObject);

      std::vector<std::shared_ptr<Object> >& getObjects() {
        return object;
      }
      
      /** Plot a separate xml and h5 file for this group if truee */
      void setSeparateFile(bool sepFile) { separateFile=sepFile; }

      bool getSeparateFile() { return separateFile; }
      std::shared_ptr<H5::File>& getHDF5File() { return hdf5File; }

      /** Returns the file name of the .ombvx file of this separateFile Group
       * including the absolute or relatvie path */
      std::string getFileName() { return fileName; }

      std::string getFullName(bool includingFileName=false, bool stopAtSeparateFile=false) override;
      
      /** Sets the file name of the .ombvx file of this separateFile Group
       * including the absolute or relatvie path */
      void setFileName(const std::string &fn) { fileName=fn; }
      
      /** Initialize/Write the tree (XML and h5).
       * Call this function for the root node of the tree to init the h5 file.
       * (Only the h5 tree is written, but do data. Use append() to write date to h5 file after calling this function)
       */
      void write(bool writeXMLFile=true, bool writeH5File=true);

      /** Read the tree (XML and h5). */
      void read();

      /** Enable SWMR if a H5 file is written. */
      void enableSWMR();

      /** Set the callback which is called, by HDF5Serie, if reading this file should be closed (and reopened immediately after) */
      void setCloseRequestCallback(const std::function<void()> &closeRequestCallback_) { closeRequestCallback=closeRequestCallback_; }

      /** terminate the tree.
       * Call this function for the root node of the free after all writing has done.
       */
      void terminate() override;

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent) override;

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
