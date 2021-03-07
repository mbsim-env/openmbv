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
      std::string getRelPathTo(const std::shared_ptr<Body> &destBody);
    protected:
      std::string outLineStr, shilouetteEdgeStr;
      DrawStyle drawMethod{filled};
      double pointSize{0};
      double lineWidth{0};
      void createHDF5File() override;
      void openHDF5File() override;
      Body();
      ~Body() override = default;
    public:
      /** Draw outline of this object in the viewer if true (the default) */
      void setOutLine(bool ol) { outLineStr=(ol)?"true":"false"; }

      bool getOutLine() { return outLineStr=="true"?true:false; }

      /** Draw shilouette edges of this object in the viewer if true (the default) */
      void setShilouetteEdge(bool ol) { shilouetteEdgeStr=(ol)?"true":"false"; }

      bool getShilouetteEdge() { return shilouetteEdgeStr=="true"?true:false; }

      /** Draw method/style of this object in the viewer (default: filled) */
      void setDrawMethod(DrawStyle ds) { drawMethod=ds; }

      DrawStyle getDrawMethod() { return drawMethod; }

      /** Point size of this object in the viewer (default: 0) */
      void setPointSize(double ps) { pointSize=ps; }

      double getPointSize() { return pointSize; }

      /** Line width of this object in the viewer (default: 0) */
      void setLineWidth(double lw) { lineWidth=lw; }

      double getLineWidth() { return lineWidth; }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent) override;

      /** Get the number of rows of the default data.
       * Returns 0, if no default data is avaliable.
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

