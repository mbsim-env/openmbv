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

#ifndef _OPENMBV_POINTSET_H
#define _OPENMBV_POINTSET_H

#include <openmbvcppinterface/rigidbody.h>
#include <vector>

namespace OpenMBV {

  /** A point set */
  class PointSet : public RigidBody {
    friend class ObjectFactory;
    protected:
      std::vector<std::vector<double> > vp;
      PointSet() = default;
      ~PointSet() override = default;
    public:
      /** Get vertex positions
       */
      const std::vector<std::vector<double> >& getVertexPositions() { return vp; }

      /** Set vertex positions
       */
      void setVertexPositions(const std::vector<std::vector<double> > &vp_) { vp = vp_; }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      xercesc::DOMElement *writeXMLFile(xercesc::DOMNode *parent) override;
  };

}

#endif
