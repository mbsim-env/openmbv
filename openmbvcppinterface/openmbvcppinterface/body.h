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

#ifndef _OPENMBV_BODY_H_
#define _OPENMBV_BODY_H_

#include <string>
#include <sstream>
#include <openmbvcppinterface/object.h>

namespace OpenMBV {

  /** Abstract base class for all bodies */
  class Body : public Object {
    private:
      std::string getRelPathTo(Body* destBody);
    protected:
      Body* hdf5LinkBody;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
      void terminate();
    public:
      /** Default constructor */
      Body();

      /** Link this body with dest in the HDF5 file */
      void setHDF5LinkTarget(Body* dest) { hdf5LinkBody=dest; }

      /** Returns if this body is linked to another */
      bool isHDF5Link() { return hdf5LinkBody!=0; }

      // FROM NOW ONLY CONVENIENCE FUNCTIONS FOLLOW !!!
    protected:
      static std::string numtostr(int i) { std::ostringstream oss; oss << i; return oss.str(); }
      static std::string numtostr(double d) { std::ostringstream oss; oss << d; return oss.str(); } 
  };

}

#endif /* _OPENMBV_BODY_H_ */

