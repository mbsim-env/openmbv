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
#include <H5Cpp.h>
#include <sstream>
#include <mbxmlutilstinyxml/tinyxml.h>
#include <mbxmlutilstinyxml/tinynamespace.h>
#include <vector>

#define OPENMBVNS_ "http://openmbv.berlios.de/OpenMBV"
#define OPENMBVNS "{"OPENMBVNS_"}"

#define MBXMLUTILSPARAMNS_ "http://openmbv.berlios.de/MBXMLUtils/parameter"
#define MBXMLUTILSPARAMNS "{"MBXMLUTILSPARAMNS_"}"

namespace OpenMBV {

  class Group;

  template<class t>
  class SimpleParameter;
  typedef SimpleParameter<double> ScalarParameter;
  typedef SimpleParameter<std::vector<double> > VectorParameter;
  typedef SimpleParameter<std::vector<std::vector<double> > > MatrixParameter;

  /** Abstract base class */
  class Object {
    friend class Group;
    protected:
      std::string name;
      std::string enableStr, boundingBoxStr;
      std::string ID; // Note: the ID is metadata and stored as a processing instruction in XML
      bool selected; // Note: the selected flag is metadata and not stored in XML but used by OpenMBVGUI
      Group* parent;

      virtual void createHDF5File()=0;
      virtual void openHDF5File()=0;
      H5::Group *hdf5Group;
      virtual void terminate()=0;

      std::map<std::string, double> scalarParameter;
      std::map<std::string, std::vector<double> > vectorParameter;
      std::map<std::string, std::vector<std::vector<double> > > matrixParameter;

      double get(const ScalarParameter& src);
      std::vector<double> get(const VectorParameter& src);
      std::vector<std::vector<double> > get(const MatrixParameter& src);

      void set(ScalarParameter& dst, const ScalarParameter& src);
      void set(VectorParameter& dst, const VectorParameter& src);
      void set(MatrixParameter& dst, const MatrixParameter& src);
      
      virtual void collectParameter(std::map<std::string, double>& sp, std::map<std::string, std::vector<double> >& vp, std::map<std::string, std::vector<std::vector<double> > >& mp, bool collectAlsoSeparateGroup=false);

      /** Virtual destructor */
      virtual ~Object();
    public:
      /** Default constructor */
      Object();

      /** It must be possible to delete the top level Group: use this function for therefore.
       * If this object is was added into a parent object this object is first removed from this parent and then deleted. */
      virtual void destroy() const;

      /** Retrun the class name */
      virtual std::string getClassName()=0;

      /** Enable this object in the viewer if true (the default) */
      void setEnable(bool enable) { enableStr=(enable==true)?"true":"false"; }

      bool getEnable() { return enableStr=="true"?true:false; }

      /** Draw bounding box of this object in the viewer if true (the default) */
      void setBoundingBox(bool bbox) { boundingBoxStr=(bbox==true)?"true":"false"; }

      bool getBoundingBox() { return boundingBoxStr=="true"?true:false; }

      /** Set the name of this object */
      void setName(const std::string& name_) { name=name_; }

      std::string getName() { return name; }

      /** Returns the full name (path) of the object */
      virtual std::string getFullName(bool includingFileName=false);

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(MBXMLUtils::TiXmlElement *element);

      virtual MBXMLUtils::TiXmlElement *writeXMLFile(MBXMLUtils::TiXmlNode *parent);

      /** return the first Group in the tree which is an separateFile */
      Group* getSeparateGroup();

      /** return the top level Group */
      Group* getTopLevelGroup();

      Group* getParent() { return parent; }

      /** get the ID sting of the Object (Note: the ID is metadata and stored as a processing instruction in XML) */
      std::string getID() const { return ID; }
      /** set the ID sting of the Object (Note: the ID is metadata and stored as a processing instruction in XML) */
      void setID(std::string ID_) { ID=ID_; }

      /** get the selected flag (Note: the selected flag is metadata and not stored in XML but used by OpenMBVGUI) */
      bool getSelected() const { return selected; }
      /** set the selected flag (Note: the selected flag is metadata and not stored in XML but used by OpenMBVGUI) */
      void setSelected(bool selected_) { selected=selected_; }

      // FROM NOW ONLY CONVENIENCE FUNCTIONS FOLLOW !!!
      static std::string fixPath(std::string oldFile, std::string newFile) { return MBXMLUtils::fixPath(oldFile, newFile); }

      static ScalarParameter getDouble(MBXMLUtils::TiXmlElement *e);
      static VectorParameter getVec(MBXMLUtils::TiXmlElement *e, unsigned int rows=0);
      static MatrixParameter getMat(MBXMLUtils::TiXmlElement *e, unsigned int rows=0, unsigned int cols=0);

      static std::string numtostr(int i) { std::ostringstream oss; oss << i; return oss.str(); }
      static std::string numtostr(double d) { std::ostringstream oss; oss << d; return oss.str(); } 


      template <class T>
      static void addElementText(MBXMLUtils::TiXmlElement *parent, std::string name, T value) {
        std::ostringstream oss;
        oss<<value;
        parent->LinkEndChild(new MBXMLUtils::TiXmlElement(name))->LinkEndChild(new MBXMLUtils::TiXmlText(oss.str()));
      }

      void addElementText(MBXMLUtils::TiXmlElement *parent, std::string name, double value, double def);

      void addElementText(MBXMLUtils::TiXmlElement *parent, std::string name, SimpleParameter<double> value, double def);

      template <class T>
      static void addAttribute(MBXMLUtils::TiXmlNode *node, std::string name, T value) {
        if(node->ToElement()) {
          std::ostringstream oss;
          oss<<value;
          node->ToElement()->SetAttribute(name, oss.str());
        }
      }

      template <class T>
      static void addAttribute(MBXMLUtils::TiXmlNode *node, std::string name, T value, std::string def) {
        if(value!=def) addAttribute(node, name, value);
      }

    protected:
      static std::vector<double> toVector(std::string str);
      static std::vector<std::vector<double> > toMatrix(std::string str);
  };

}

#endif
