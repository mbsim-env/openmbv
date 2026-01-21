/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

  This library is free software; you can redistribute it and/or 
  modify it under the terms of the GNU Lesser General Public 
  License as published by the Free Software Foundation; either 
  version 2.1 of the License, or (at your option) any later version. 
   
  This library is distributed in the hope that it will be useful, 
  but WITHOUT ANY WARRANTY; without even the implied warranty of 
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
  Lesser General Public License for more details. 
   
  You should have received a copy of the GNU Lesser General Public 
  License along with this library; if not, write to the Free Software 
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
*/

#ifndef _OPENMBV_GROUP_H_
#define _OPENMBV_GROUP_H_

#include <openmbvcppinterface/objectfactory.h>
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
      boost::filesystem::path fileName; // the file name of the .ombvx file including the absolute or relatvie path
      std::shared_ptr<H5::File> hdf5File;
      std::function<void()> closeRequestCallback;
      std::function<void()> refreshCallback;
      void createHDF5File() override;
      void openHDF5File() override;

      Group();
      ~Group() override = default;

      std::shared_ptr<xercesc::DOMDocument> writeXMLDoc();
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
      
      std::shared_ptr<H5::File>& getHDF5File() { return hdf5File; }

      /** Returns the file name of the .ombvx file including the absolute or relatvie path */
      std::string getFileName() { return fileName.string(); }

      std::string getFullName() override;
      
      /** Sets the file name of the .ombvx file including the absolute or relatvie path */
      void setFileName(const boost::filesystem::path &fn) { fileName=fn; }
      
      /** Initialize/Write the tree (XML and h5).
       * Call this function for the root node of the tree to init the h5 file.
       * (Only the h5 tree is written, but do data. Use append() to write date to h5 file after calling this function)
       */
      void write(bool writeXMLFile=true, bool writeH5File=true, bool embedXMLInH5=false);

      /** Read the tree (XML and h5). */
      void read();

      /** Enable SWMR if a H5 file is written. */
      void enableSWMR();

      /** Flush the H5 file if reader has requested a flush. */
      void flushIfRequested();

      /** Refresh the H5 file. */
      void refresh();

      /** Request a flush of the writer.
       * If a writer process currently exists true is returned else false. Note that this flag cannot change
       * while the calling process as opened the file for reading.
       */
      bool requestFlush();

      /** Set the callback which is called, by HDF5Serie, if reading this file should be closed (and reopened immediately after) */
      void setCloseRequestCallback(const std::function<void()> &closeRequestCallback_) { closeRequestCallback=closeRequestCallback_; }

      /** Set the callback which is called, by HDF5Serie, if, after a writer flush, the writer has finished the flush
       * and this reader should now refresh the file. */
      void setRefreshCallback(const std::function<void()> &refreshCallback_) { refreshCallback=refreshCallback_; }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent) override;

      /** return the top level Group */
      std::shared_ptr<Group> getTopLevelGroup() {
        std::shared_ptr<Group> p=parent.lock();
        return !p?shared_from_this():p->getTopLevelGroup();
      }
  };

}

#endif
