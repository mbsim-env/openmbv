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

#ifndef _OPENMBV_DYNAMICCOLOREDBODY_H_
#define _OPENMBV_DYNAMICCOLOREDBODY_H_

#include <openmbvcppinterface/body.h>

namespace OpenMBV {

  /** Abstract base class for all dynamically colored bodies */
  class DynamicColoredBody : public Body {
    protected:
      ScalarParameter minimalColorValue, maximalColorValue;
      double dynamicColor;
      VectorParameter diffuseColor;
      ScalarParameter transparency;

      ~DynamicColoredBody();
    public:
      DynamicColoredBody();

      /** Set the minimal color value.
       * The color value of the body in linearly mapped between minimalColorValue
       * and maximalColorValue to blue(minimal) over cyan, green, yellow to red(maximal).
       */
      void setMinimalColorValue(const ScalarParameter min) {
        set(minimalColorValue,min);
      }

      double getMinimalColorValue() { return get(minimalColorValue); }

      /** Set the maximal color value.
       * See also minimalColorValue
       */
      void setMaximalColorValue(const ScalarParameter max) {
        set(maximalColorValue,max);
      }

      double getMaximalColorValue() { return get(maximalColorValue); }

      /** Set the color for the body dynamically.
       * If this value is set, the color given to the append function
       * (as last element of the data row) is overwritten with this value.
       */
      void setDynamicColor(const double col) {
        dynamicColor=col;
      }

      double getDynamicColor() { return dynamicColor; }

      /** Set the diffuse color of the body (HSV values from 0 to 1).
       * If the hue is less then 0 (default = -1) then the dynamic color from the
       * append routine is used as hue value.
       */
      void setDiffuseColor(const VectorParameter &hsv) {
        if(hsv.getParamStr()=="" && hsv.getValue().size()!=3) throw std::runtime_error("the dimension does not match");
        set(diffuseColor,hsv);
      }

      void setDiffuseColor(const std::vector<double> &hsv) {
        if(hsv.size()!=3) throw std::runtime_error("the diemension does not match");
        set(diffuseColor,hsv);
      }

      void setDiffuseColor(double h, double s, double v) {
        std::vector<double> hsv;
        hsv.push_back(h);
        hsv.push_back(s);
        hsv.push_back(v);
        diffuseColor=hsv;
      }

      std::vector<double> getDiffuseColor() { return get(diffuseColor); }

      /** Set the transparency of the body. */
      void setTransparency(ScalarParameter t) {
        set(transparency,t);
      }

      double getTransparency() { return get(transparency); }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(xercesc::DOMElement *element);

      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent);
  };

}

#endif
