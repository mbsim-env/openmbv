#ifndef _OPENMBV_OBJECT_H_
#define _OPENMBV_OBJECT_H_

#include <string>
#include <H5Cpp.h>

#define OPENMBVNS_ "http://openmbv.berlios.de/OpenMBV"

namespace OpenMBV {

  class Group;

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
      Object();
      virtual ~Object();
      void setName(const std::string& name_) { name=name_; }
      void setExpand(bool expand) { expandStr=(expand==true)?"true":"false"; }
      std::string getFullName();
  };

}

#endif
