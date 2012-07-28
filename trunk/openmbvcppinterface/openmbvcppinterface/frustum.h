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

#ifndef _OPENMBV_FRUSTUM_H_
#define _OPENMBV_FRUSTUM_H_

#include <openmbvcppinterface/rigidbody.h>

namespace OpenMBV {

  /** A frustum (with a frustum hole) */
  class Frustum : public RigidBody {
    protected:
      ScalarParameter baseRadius, topRadius, height, innerBaseRadius, innerTopRadius;
      ~Frustum() {}
    public:
      /** Default constructor */
      Frustum();

      /** Retrun the class name */
      std::string getClassName() { return "Frustum"; }

      /** Set the radius of the outer side at the base (bottom) */
      void setBaseRadius(ScalarParameter radius) {
        set(baseRadius,radius);
      } 

      double getBaseRadius() { return get(baseRadius); }

      /** Set the radius of the outer side at the top. */
      void setTopRadius(ScalarParameter radius) {
        set(topRadius,radius);
      } 

      double getTopRadius() { return get(topRadius); }

      /** Set height of the frustum */
      void setHeight(ScalarParameter height_) {
        set(height,height_);
      } 

      double getHeight() { return get(height); }

      /** Set the radius of the inner side at the base (bottom). */
      void setInnerBaseRadius(ScalarParameter radius) {
        set(innerBaseRadius,radius);
      } 

      double getInnerBaseRadius() { return get(innerBaseRadius); }

      /** Set the radius of the inner side at the top. */
      void setInnerTopRadius(ScalarParameter radius) {
        set(innerTopRadius,radius);
      } 

      double getInnerTopRadius() { return get(innerTopRadius); }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(TiXmlElement *element);

      TiXmlElement* writeXMLFile(TiXmlNode *parent);
  };

}

#endif
