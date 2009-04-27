#ifndef _OPENMBV_GROUP_H_
#define _OPENMBV_GROUP_H_

#include <openmbvcppinterface/object.h>
#include <vector>
#include <H5Cpp.h>

namespace OpenMBV {

  /** A container for bodies */
  class Group : public Object {
    friend class Body;
    protected:
      std::vector<Object*> object;
      bool separateFile;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
    public:
      /** Default constructor */
      Group();

      /** Add a object to this object container */
      void addObject(Object* object);
      
      /** Plot a separate xml and h5 file for this group if truee */
      void setSeparateFile(bool sepFile) { separateFile=sepFile; }
      
      /** Initialisze the tree.
       * Call this function for the root node of the tree before starting writing.
       */
      void initialize();

      /** terminate the tree.
       * Call this function for the root node of the free after all writing has done.
       */
      void terminate();
  };

}

#endif
