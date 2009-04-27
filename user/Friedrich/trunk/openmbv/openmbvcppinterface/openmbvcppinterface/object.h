#ifndef _OPENMBV_OBJECT_H_
#define _OPENMBV_OBJECT_H_

#include <string>
#include <H5Cpp.h>

#define OPENMBVNS_ "http://openmbv.berlios.de/OpenMBV"

namespace OpenMBV {

  class Group;

  /** Abstract base class */
  class Object {
    friend class Group;
    protected:
      std::string name;
      std::string expandStr;
      Group* parent;
      virtual void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="")=0;
      virtual void createHDF5File()=0;
      H5::Group *hdf5Group;
      virtual void terminate()=0;
    public:
      /** Default constructor */
      Object();

      /** Virtual destructor */
      virtual ~Object();

      /** Set the name of this object */
      void setName(const std::string& name_) { name=name_; }

      /** Expand this tree node in a view if true (the default) */
      void setExpand(bool expand) { expandStr=(expand==true)?"true":"false"; }

      /** Returns the full name (path) of the object */
      std::string getFullName();
  };

}

#endif
