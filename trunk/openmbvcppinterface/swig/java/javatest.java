// You have to add /your/MBSim/install/path/bin/openmbv.jar to your classpath when compiling or running this program.

import de.berlios.openmbv.OpenMBV.*;

public class javatest {

  // main program
  public static void main(String argv[]) {
    // create main group
    Group group=new Group();
    group.setName("MBS");

    // add some objects to group in this subroutine and return a added cube
    Cube cube=createMyGroup(group);
    
    // create H5 and xml file
    group.setFileName("MBS_outfile.ombv.xml");
    group.write(true, true);
    
    // some action on the returned cube
    String ret=cube.getName(); System.out.println(ret);
    boolean b=cube.getReferenceFrame(); System.out.println(b);
    double d=cube.getLength(); System.out.println(d);
    cube.append(new double[]{0.4, 1, 2, 3, 4, 5, 6, 0.25});
  }


  // a subroutine
  public static Cube createMyGroup(Group g) {
    // create a cube
    Cube cube=new Cube();
    cube.setName("Box1");
    cube.setReferenceFrame(true);
    cube.setLength(1.234);
    //cube.setLength(new ScalarParameter(1.235)); ScalarParameter is not supported by java
    //cube.setLength(new ScalarParameter("p1", 1.236)); ScalarParameter is not supported by java
    
    // create a cuboid
    Cuboid cuboid=new Cuboid();
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
