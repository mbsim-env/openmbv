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

#ifndef _OPENMBV_COMPOUNDRIGIDBODY_H_
#define _OPENMBV_COMPOUNDRIGIDBODY_H_

#include <openmbvcppinterface/rigidbody.h>
#include <vector>

namespace OpenMBV {

  /** A compound of rigid bodies */
  class CompoundRigidBody : public RigidBody
#ifndef SWIG
      , public std::enable_shared_from_this<CompoundRigidBody>
#endif
    {
    friend class RigidBody;
    friend class ObjectFactory;
    protected:
      std::string expandStr;
      std::vector<std::shared_ptr<RigidBody> > rigidBody;

      CompoundRigidBody();
      ~CompoundRigidBody() override;
    public:
      /** Add a RigidBody to this compound */
      void addRigidBody(const std::shared_ptr<RigidBody>& rigidBody_) {
        if(rigidBody_->name.empty()) throw std::runtime_error("the object to be added must have a name");
        for(auto & i : rigidBody)
          if(i->name==rigidBody_->name) throw std::runtime_error("a object of the same name already exists");
        rigidBody.push_back(rigidBody_);
        rigidBody_->parent.reset();
        rigidBody_->compound=shared_from_this();
      }

      std::vector<std::shared_ptr<RigidBody> >& getRigidBodies() {
        return rigidBody;
      }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent) override;

      /** Expand this tree node in a view if true (the default) */
      void setExpand(bool expand) { expandStr=(expand)?"true":"false"; }

      bool getExpand() { return expandStr=="true"?true:false; }
  };

}

#endif
