/* Copyright (C) 2004-2010 OpenMBV Development Team
 *
 * This library is free software; you can redistribute it and/or 
 * modify it under the terms of the GNU Lesser General Public 
 * License as published by the Free Software Foundation; either 
 * version 2.1 of the License, or (at your option) any later version. 
 *  
 * This library is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
 * Lesser General Public License for more details. 
 *  
 * You should have received a copy of the GNU Lesser General Public 
 * License along with this library; if not, write to the Free Software 
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
 *
 * Contact: friedrich.at.gc@googlemail.com
 */

#ifndef _OPENMBV_OBJECTFACTORY_H_
#define _OPENMBV_OBJECTFACTORY_H_

#include <map>
#include <stdexcept>
#include <typeinfo>
#include <mbxmlutilshelper/dom.h>
#include <xercesc/dom/DOMElement.hpp>

namespace OpenMBV {

class Object;

/** A object factory.
 * A object facroty which creates any object derived from Object.
 */
class ObjectFactory {

  public:

    /** Register the class CreateType which the XML element name name by the object factory.
     * You should not use this function directly but
     * see also the macro OPENMBV_OBJECTFACTORY_REGISTERXMLNAME.  */
    template<class CreateType>
    static void registerXMLName(const MBXMLUtils::FQN &name) {
      registerXMLName(name, &allocate<CreateType>);
    }

    /** Create an object corresponding to the XML element element and return a pointer of type ContainerType.
     * Throws if the created object is not of type ContainerType.
     * This function returns a new object dependent on the registration of the created object. */
    template<class ContainerType>
    static std::shared_ptr<ContainerType> create(const xercesc::DOMElement *element) {
      // just check if ContainerType is derived from Object if not throw a compile error
      static_assert(std::is_convertible<ContainerType*, Object*>::value,
        "In OpenMBV::ObjectFactory::create<ContainerType>(...) ContainerType must be derived from Object.");
      // return NULL if no input is supplied
      if(element==nullptr) return std::shared_ptr<ContainerType>();
      // loop over all all registred types corresponding to element->ValueStr()
      std::pair<MapIt, MapIt> range=instance().registeredType.equal_range(MBXMLUtils::E(element)->getTagName());
      for(auto it=range.first; it!=range.second; it++) {
        // allocate a new object using the allocate function pointer
        std::shared_ptr<Object> ele=it->second();
        // try to cast ele up to ContainerType
        std::shared_ptr<ContainerType> ret=std::dynamic_pointer_cast<ContainerType>(ele);
        // if possible, return it
        if(ret)
          return ret;
      }
      // no matching element found: throw error
      throw std::runtime_error("No class named "+MBXMLUtils::X()%element->getTagName()+" found which is of type "+
                               typeid(ContainerType).name()+".");
    }

    /** Create an empty object of type CreateType. */
    template<class CreateType>
    static std::shared_ptr<CreateType> create() {
      return std::shared_ptr<CreateType>(new CreateType, &deleter<CreateType>);
    }

    /** Return a copy of t. */
    template<class CreateType>
    static std::shared_ptr<CreateType> create(const std::shared_ptr<CreateType> &t) {
      return std::shared_ptr<CreateType>(new CreateType(*t.get()), &deleter<CreateType>);
    }

    static void addErrorMsg(const std::string &msg);
    static std::string getAndClearErrorMsg();

  private:

    // a pointer to a function allocating an object
    typedef std::shared_ptr<Object> (*allocateFkt)();

    // convinence typedefs
    typedef std::multimap<MBXMLUtils::FQN, allocateFkt> Map;
    typedef typename Map::iterator MapIt;

    // private ctor
    ObjectFactory() = default;

    static void registerXMLName(const MBXMLUtils::FQN &name, allocateFkt alloc);

    // create an singleton instance of the object factory.
    // only declaration here and defition and explicit instantation for all Object in objectfactory.cc (required for Windows)
    static ObjectFactory& instance();

    // a multimap of all registered types
    Map registeredType;

    // a wrapper to allocate an object of type CreateType: used by create(xercesc::DOMElement *)
    template<class CreateType>
    static std::shared_ptr<Object> allocate() {
      return std::shared_ptr<CreateType>(new CreateType, &deleter<CreateType>);
    }

    // a wrapper to deallocate an object of type T: all dtors are protected but ObjectFactory is a friend of all classes
    template<class T>
    static void deleter(T *t) { delete t; }

    static std::string errorMsg;
};

/** Helper function for automatic class registration for ObjectFactory.
 * You should not use this class directly but
 * use the macro OPENMBV_REGISTER_XMLNAME_AT_OBJECTFACTORY. */
template<class CreateType>
class ObjectFactoryRegisterXMLNameHelper {

  public:

    /** ctor registring the new type */
    ObjectFactoryRegisterXMLNameHelper(const MBXMLUtils::FQN &name) noexcept {
      try {
        ObjectFactory::registerXMLName<CreateType>(name);
      }
      catch(std::exception &ex) {
        ObjectFactory::addErrorMsg(ex.what());
      }
      catch(...) {
        ObjectFactory::addErrorMsg("Unknown error");
      }
    };

};

}

/** Use this macro somewhere at the class definition of ThisType to register it by the ObjectFactory.
 * ThisType must have a public default ctor and a public dtor. */
#define OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(ThisType, name) \
  static OpenMBV::ObjectFactoryRegisterXMLNameHelper<ThisType> BOOST_PP_CAT(objectFactoryRegistrationDummyVariable_, __LINE__)(name);


#endif
