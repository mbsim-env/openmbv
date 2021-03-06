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

#ifndef _OPENMBV_DYNAMICINDEXEDLINESET_H
#define _OPENMBV_DYNAMICINDEXEDLINESET_H

#include <openmbvcppinterface/flexiblebody.h>
#include <vector>
#include <hdf5serie/vectorserie.h>

namespace OpenMBV {

  /** A dynamic indexed line set */
  class DynamicIndexedLineSet : public FlexibleBody {
    friend class ObjectFactory;
    protected:
      DynamicIndexedLineSet() = default;
      ~DynamicIndexedLineSet() override = default;
      xercesc::DOMElement *writeXMLFile(xercesc::DOMNode *parent) override;
      std::vector<Index> indices;

    public:
      /** Get indices
       */
      const std::vector<Index>& getIndices() { return indices; }

      /** Set indices
       */
      void setIndices(const std::vector<Index> &indices_) { indices = indices_; }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;
  };

}

#endif
