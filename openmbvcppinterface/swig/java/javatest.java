// You have to add /your/MBSim/install/path/bin/openmbv.jar to your classpath when compiling or running this program.

import de.mbsim_env.openmbv.*;

public class javatest {

  // main program
  public static void main(String argv[]) {
    mymain();
  }

  public static void mymain() {
    // create main group
    Group group=ObjectFactory.create_Group();
    group.setName("MBS");

    // add some objects to group in this subroutine and return a added cube
    Cube cube=createMyGroup(group);

    // add a IndexedFaceSet
    IndexedFaceSet ifs=ObjectFactory.create_IndexedFaceSet();
    ifs.setName("IFS");
    int indices[]=new int[]{2, 6, 3, 1};
    for(int i=0; i<indices.length; ++i)
      System.out.println(indices[i]);
    ifs.setIndices(indices);
    group.addObject(ifs);
    
    // create H5 and xml file
    group.setFileName("MBS_outfile.ombvx");
    group.write(true, true);
    
    // some action on the returned cube
    String ret=cube.getName(); System.out.println(ret);
    boolean b=cube.getReferenceFrame(); System.out.println(b);
    double d=cube.getLength(); System.out.println(d);
    cube.append(new double[]{0.4, 1, 2, 3, 4, 5, 6, 0.25});

    // You need to call delete to call the dtor of group (which closes the file)
    // Note that java may even never call the dtor without this statement since its even not guarantieed
    // that java runs the garbage collector and finalizer at program end
    group.delete();
  }


  // a subroutine
  public static Cube createMyGroup(Group g) {
    // create a cube
    Cube cube=ObjectFactory.create_Cube();
    cube.setName("Box1");
    cube.setReferenceFrame(true);
    cube.setLength(1.234);
    
    // create a cuboid
    Cuboid cuboid=ObjectFactory.create_Cuboid();
    cuboid.setName("Box2");
    cuboid.setReferenceFrame(true);
    cuboid.setLength(1.234, 3, 4);
    
    // add cube and cuboid to group
    g.addObject(cube);
    g.addObject(cuboid);
    
    // some actions on cuboid
    double[] dv=cuboid.getLength(); System.out.println(dv[0]); System.out.println(dv[1]); System.out.println(dv[2]);
    cuboid.setLength(new double[]{5.6, 9.7, 3.5});
    dv=cuboid.getLength(); System.out.println(dv[0]); System.out.println(dv[1]); System.out.println(dv[2]);

    // return the cube
    return cube;
  }
}
