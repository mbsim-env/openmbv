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

namespace OpenMBV {

  /** A container for bodies */
  class Group : public Object {
    friend class Body;
    protected:
      std::vector<Object*> object;
      std::string expandStr;
      std::string fileName;
      bool separateFile, topLevelFile;
      TiXmlElement* writeXMLFile(TiXmlNode *parent);
      void createHDF5File();
      void openHDF5File();
      void readSimpleParameter(std::string filename);
      void writeSimpleParameter(std::string filename);
      void collectParameter(std::map<std::string, double>& sp, std::map<std::string, std::vector<double> >& vp, std::map<std::string, std::vector<std::vector<double> > >& mp, bool collectAlsoSeparateGroup=false);

      virtual ~Group();
    public:
      /** Default constructor */
      Group();

      /** It must be possible to delete the top level Group: use this function for therefore.
       * If this object is was added into a parent object this object is first removed from this parent and then deleted. */
      void destroy() const;

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

      void setTopLevelFile(bool b) { topLevelFile=b; }
      bool getTopLevelFile() { return topLevelFile; }

      // fileName is only set if separateFile is true
      std::string getFileName() { return fileName; }
      
      /** Initialisze/Write the XML file.
       * Call this function for the root node of the tree to create/write/ the XML file.
       */
      void writeXML();

      /** Read the XML file.
       * Call this function to read an OpenMBV XML file and creating the Object tree.
       */
      static Group* readXML(std::string fileName);
      
      /** Initialisze/Write the h5 file.
       * Call this function for the root node of the tree to init the h5 file.
       * (Only the h5 tree i written, but do data. Use append() to write date to h5 file after calling this function)
       */
      void writeH5();

      /** Read/open an existing h5 file. Before calling this function readXML() must be called or simply call read().
       */
      static void readH5(Group *rootGrp);

      /** Initialisze/Wrtie the tree (XML and h5).
       * This function simply calls writeXML() and writeH5().
       */
      void write() { writeXML(); writeH5(); }

      /** Read the tree (XML and h5).
       * This function simply calls readXML() and readH5().
       */
      static Group* read(const std::string& filename) { Group *rootGrp=readXML(filename); readH5(rootGrp); return rootGrp; }

      /** terminate the tree.
       * Call this function for the root node of the free after all writing has done.
       */
      void terminate();

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(TiXmlElement *element);

      /** return the first Group in the tree which is an separateFile */
      Group* getSeparateGroup() { return separateFile?this:parent->getSeparateGroup(); }

      /** return the top level Group (this Group is an topLevelFile */
      Group* getTopLevelGroup() { return topLevelFile?this:parent->getTopLevelGroup(); }

      double getScalarParameter(std::string name);
      std::vector<double> getVectorParameter(std::string name);
      std::vector<std::vector<double> > getMatrixParameter(std::string name);
  };

}

#endif
