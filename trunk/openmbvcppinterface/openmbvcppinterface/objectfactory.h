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
#ifdef HAVE_BOOST_TYPE_TRAITS_HPP
# include <boost/static_assert.hpp>
# include <boost/type_traits.hpp>
#endif

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
      registerXMLName(name, &allocate<CreateType>, &deallocate);
    }

    /** Create an object corresponding to the XML element element and return a pointer of type ContainerType.
     * Throws if the created object is not of type ContainerType.
     * This function returns a new object dependent on the registration of the created object. */
    template<class ContainerType>
    static ContainerType* create(const xercesc::DOMElement *element) {
#ifdef HAVE_BOOST_TYPE_TRAITS_HPP
      // just check if ContainerType is derived from Object if not throw a compile error if boost is avaliable
      // if boost is not avaliable a runtime error will occure later. (so it does not care if boost is not available)
      BOOST_STATIC_ASSERT_MSG((boost::is_convertible<ContainerType*, Object*>::value),
        "In OpenMBV::ObjectFactory::create<ContainerType>(...) ContainerType must be derived from Object.");
#endif
      // return NULL if no input is supplied
      if(element==NULL) return NULL;
      // loop over all all registred types corresponding to element->ValueStr()
      std::pair<MapIt, MapIt> range=instance().registeredType.equal_range(MBXMLUtils::E(element)->getTagName());
      for(MapIt it=range.first; it!=range.second; it++) {
        // allocate a new object using the allocate function pointer
        Object *ele=it->second.first();
        // try to cast ele up to ContainerType
        ContainerType *ret=dynamic_cast<ContainerType*>(ele);
        // if possible, return it
        if(ret)
          return ret;
        // if not possible, deallocate newly created (wrong) object and continue searching
        else
          it->second.second(ele);
      }
      // no matching element found: throw error
      throw std::runtime_error("No class named "+MBXMLUtils::X()%element->getTagName()+" found which is of type "+
                               typeid(ContainerType).name()+".");
    }

  private:

    // a pointer to a function allocating an object
    typedef Object* (*allocateFkt)();
    // a pointer to a function deallocating an object
    typedef void (*deallocateFkt)(Object *obj);

    // convinence typedefs
    typedef std::multimap<MBXMLUtils::FQN, std::pair<allocateFkt, deallocateFkt> > Map;
    typedef typename Map::iterator MapIt;

    // private ctor
    ObjectFactory() {}

    static void registerXMLName(const MBXMLUtils::FQN &name, allocateFkt alloc, deallocateFkt dealloc);

    // create an singleton instance of the object factory.
    // only declaration here and defition and explicit instantation for all Object in objectfactory.cc (required for Windows)
    static ObjectFactory& instance();

    // a multimap of all registered types
    Map registeredType;

    // a wrapper to allocate an object of type CreateType
    template<class CreateType>
    static Object* allocate() {
      return new CreateType;
    }

    // a wrapper to deallocate an object created by allocate
    static void deallocate(Object *obj);

};

/** Helper function for automatic class registration for ObjectFactory.
 * You should not use this class directly but
 * use the macro OPENMBV_REGISTER_XMLNAME_AT_OBJECTFACTORY. */
template<class CreateType>
class ObjectFactoryRegisterXMLNameHelper {

  public:

    /** ctor registring the new type */
    ObjectFactoryRegisterXMLNameHelper(const MBXMLUtils::FQN &name) {
      ObjectFactory::registerXMLName<CreateType>(name);
    };

};

}

#define OPENMBV_OBJECTFACTORY_CONCAT1(X, Y) X##Y
#define OPENMBV_OBJECTFACTORY_CONCAT(X, Y) OPENMBV_OBJECTFACTORY_CONCAT1(X, Y)
#define OPENMBV_OBJECTFACTORY_APPENDLINE(X) OPENMBV_OBJECTFACTORY_CONCAT(X, __LINE__)

/** Use this macro somewhere at the class definition of ThisType to register it by the ObjectFactory.
 * ThisType must have a public default ctor and a public dtor. */
#define OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(ThisType, name) \
  static OpenMBV::ObjectFactoryRegisterXMLNameHelper<ThisType> \
    OPENMBV_OBJECTFACTORY_APPENDLINE(objectFactoryRegistrationDummyVariable)(name);

#endif
