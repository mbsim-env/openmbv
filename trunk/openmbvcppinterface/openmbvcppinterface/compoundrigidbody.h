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

#ifndef _OPENMBV_COMPOUNDRIGIDBODY_H_
#define _OPENMBV_COMPOUNDRIGIDBODY_H_

#include <openmbvcppinterface/rigidbody.h>
#include <vector>

namespace OpenMBV {

  /** A compound of rigid bodies */
  class CompoundRigidBody : public RigidBody {
    protected:
      std::string expandStr;
      std::vector<RigidBody*> rigidBody;
      TiXmlElement* writeXMLFile(TiXmlNode *parent);
      void collectParameter(std::map<std::string, double>& sp, std::map<std::string, std::vector<double> >& vp, std::map<std::string, std::vector<std::vector<double> > >& mp, bool collectAlsoSeparateGroup=false);

      ~CompoundRigidBody();
    public:
      /** Default constructor */
      CompoundRigidBody();

      /** Retrun the class name */
      std::string getClassName() { return "CompoundRigidBody"; }

      /** Add a RigidBody to this compound */
      void addRigidBody(RigidBody* rigidBody_) {
        assert(rigidBody_->name!="");
        for(unsigned int i=0; i<rigidBody.size(); i++)
          assert(rigidBody[i]->name!=rigidBody_->name);
        rigidBody.push_back(rigidBody_);
        rigidBody_->parent=0;
        rigidBody_->compound=this;
      }

      std::vector<RigidBody*> getRigidBodies() {
        return rigidBody;
      }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(TiXmlElement *element);

      /** Expand this tree node in a view if true (the default) */
      void setExpand(bool expand) { expandStr=(expand==true)?"true":"false"; }

      bool getExpand() { return expandStr=="true"?true:false; }
  };

}

#endif
