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
      double minimalColorValue, maximalColorValue;
      double staticColor;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
    public:
      DynamicColoredBody();

      ~DynamicColoredBody();

      /** Set the minimal color value.
       * The color value of the body in linearly mapped between minimalColorValue
       * and maximalColorValue to blue(minimal) over cyan, green, yellow to red(maximal).
       */
      void setMinimalColorValue(const double min) {
        minimalColorValue=min;
      }

      /** Set the maximal color value.
       * See also minimalColorValue
       */
      void setMaximalColorValue(const double max) {
        maximalColorValue=max;
      }

      /** Set a static color for the body.
       * If this value is set, the color given to the append function
       * (as last element of the data row) is overwritten with this value.
       */
      void setStaticColor(const double col) {
        staticColor=col;
      }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(TiXmlElement *element);
  };

}

#endif
