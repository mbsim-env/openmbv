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
#include <H5Cpp.h>
#ifdef HAVE_BOOST_FILE_LOCK
#  include <boost/interprocess/sync/file_lock.hpp>
#endif

namespace OpenMBV {

  /** A container for bodies */
  class Group : public Object {
    friend class Body;
    friend class Object;
    protected:
      std::vector<Object*> object;
      std::string expandStr;
      std::string fileName; // the file name of the .ombv.xml file of this separateFile Group including the absolute or relatvie path
      bool separateFile;
      TiXmlElement* writeXMLFile(TiXmlNode *parent);
      void createHDF5File();
      void openHDF5File();
      void readSimpleParameter();
      void writeSimpleParameter();
      void collectParameter(std::map<std::string, double>& sp, std::map<std::string, std::vector<double> >& vp, std::map<std::string, std::vector<std::vector<double> > >& mp, bool collectAlsoSeparateGroup=false);

      virtual ~Group();

#ifdef HAVE_BOOST_FILE_LOCK
      // used for locking
      FILE *lockFile; // dummy file used for locking
      bool lockedExclusive; // true = write locked; false = read locked
      boost::interprocess::file_lock *lockFileLock;
#endif

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

      void lock(bool exclusive=false);
      void unlock();

    public:
      /** Default constructor */
      Group();

      /** Retrun the class name */
      std::string getClassName() { return "Group"; }

      /** Expand this tree node in a view if true (the default) */
      void setExpand(bool expand) { expandStr=(expand==true)?"true":"false"; }

      bool getExpand() { return expandStr=="true"?true:false; }

      /** Add a object to this object container */
      void addObject(Object* object);

      std::vector<Object*> getObjects() {
        return object;
      }
      
      /** Plot a separate xml and h5 file for this group if truee */
      void setSeparateFile(bool sepFile) { separateFile=sepFile; }

      bool getSeparateFile() { return separateFile; }

      /** Returns the file name of the .ombv.xml file of this separateFile Group
       * including the absolute or relatvie path */
      std::string getFileName() { return fileName; }
      
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
      virtual void initializeUsingXML(TiXmlElement *element);

      /** return the first Group in the tree which is an separateFile */
      Group* getSeparateGroup() { return separateFile?this:parent->getSeparateGroup(); }

      /** return the top level Group */
      Group* getTopLevelGroup() { return parent==NULL?this:parent->getTopLevelGroup(); }

      double getScalarParameter(std::string name);
      std::vector<double> getVectorParameter(std::string name);
      std::vector<std::vector<double> > getMatrixParameter(std::string name);
  };

}

#endif
