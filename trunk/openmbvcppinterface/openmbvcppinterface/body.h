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

#ifndef _OPENMBV_BODY_H_
#define _OPENMBV_BODY_H_

#include <string>
#include <sstream>
#include <vector>
#include <openmbvcppinterface/object.h>

namespace OpenMBV {

  /** Abstract base class for all bodies */
  class Body : public Object {
    public:
      enum DrawStyle { filled, lines, points };
    private:
      std::string getRelPathTo(const boost::shared_ptr<Body> &destBody);
    protected:
      std::string outLineStr, shilouetteEdgeStr;
      DrawStyle drawMethod;
      boost::shared_ptr<Body> hdf5LinkBody;
      std::string hdf5LinkStr;
      void createHDF5File();
      void openHDF5File();
      void terminate();
      Body();
      ~Body() {}
    public:

      /** Retrun the class name */
      std::string getClassName() { return "Body"; }

      /** Link this body with dest in the HDF5 file */
      void setHDF5LinkTarget(const boost::shared_ptr<Body> &dest) { hdf5LinkBody=dest; }
      
      boost::shared_ptr<Body>& getHDF5LinkTarget() { return hdf5LinkBody; }

      /** Returns if this body is linked to another */
      bool isHDF5Link() { return (hdf5LinkBody || hdf5LinkStr!=""); }

      /** Draw outline of this object in the viewer if true (the default) */
      void setOutLine(bool ol) { outLineStr=(ol==true)?"true":"false"; }

      bool getOutLine() { return outLineStr=="true"?true:false; }

      /** Draw shilouette edges of this object in the viewer if true (the default) */
      void setShilouetteEdge(bool ol) { shilouetteEdgeStr=(ol==true)?"true":"false"; }

      bool getShilouetteEdge() { return shilouetteEdgeStr=="true"?true:false; }

      /** Draw method/style of this object in the viewer (default: filled) */
      void setDrawMethod(DrawStyle ds) { drawMethod=ds; }

      DrawStyle getDrawMethod() { return drawMethod; }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(xercesc::DOMElement *element);

      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent);

      /** Pure virtual function to append a vector<double> row to the data.
       * NOTE that this is an convinience function, because most OpenMBV data is
       * of this type. If you do not need this spezial type, simple define this function
       * with empty body and create your own spezific append function
       */
      virtual void append(const std::vector<double>& row)=0;

      /** Get the number of rows of the default data.
       * Returns -1, if no default data is avaliable.
       * NOTE: see also append()
       */
      virtual int getRows()=0;

      /** Get row number i of the default data.
       * NOTE: see also append()
       */
      virtual std::vector<double> getRow(int i)=0;
  };

}

#endif /* _OPENMBV_BODY_H_ */

