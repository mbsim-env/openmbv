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

#ifndef _OPENMBV_INDEXEDLINESET_H
#define _OPENMBV_INDEXEDLINESET_H

#include <openmbvcppinterface/rigidbody.h>
#include <vector>

namespace OpenMBV {

  /** An indexed line set */
  class IndexedLineSet : public RigidBody {
    friend class ObjectFactory;
    protected:
      std::vector<std::vector<double> > vp;
      std::vector<Index> indices;
      IndexedLineSet() = default;
      ~IndexedLineSet() override = default;
    public:
      /** Get vertex positions
       */
      const std::vector<std::vector<double> >& getVertexPositions() { return vp; }
      /** Get indices
       */
      const std::vector<Index>& getIndices() { return indices; }

      /** Set vertex positions
       */
      void setVertexPositions(const std::vector<std::vector<double> > &vp_) { vp = vp_; }
      /** Set indices
       */
      void setIndices(const std::vector<Index> &indices_) { indices = indices_; }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      xercesc::DOMElement *writeXMLFile(xercesc::DOMNode *parent) override;
  };

}

#endif
