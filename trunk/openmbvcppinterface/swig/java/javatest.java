// You have to add /your/MBSim/install/path/bin/openmbv.jar to your classpath when compiling or running this program.

import de.berlios.openmbv.OpenMBV.*;

public class javatest {

  public static void main(String argv[]) {
    Group group=new Group();
    group.setName("MBS");
    
    Cube cube=new Cube();
    cube.setName("Box1");
    cube.setReferenceFrame(true);
    cube.setLength(1.234);
    //cube.setLength(new ScalarParameter(1.235)); ScalarParameter is not supported by java
    //cube.setLength(new ScalarParameter("p1", 1.236)); ScalarParameter is not supported by java
    
    Cuboid cuboid=new Cuboid();
    cuboid.setName("Box2");
    cuboid.setReferenceFrame(true);
    cuboid.setLength(1.234, 3, 4);
    
    group.addObject(cube);
    group.addObject(cuboid);
    
    group.setFileName("MBS_outfile.ombv.xml");
    group.write(true, true);
    
    String ret=cube.getName(); System.out.println(ret);
    boolean b=cube.getReferenceFrame(); System.out.println(b);
    double d=cube.getLength(); System.out.println(d);
    double[] dv=cuboid.getLength(); System.out.println(dv[0]); System.out.println(dv[1]); System.out.println(dv[2]);

    cuboid.setLength(new double[]{5.6, 9.7, 3.5});
    dv=cuboid.getLength(); System.out.println(dv[0]); System.out.println(dv[1]); System.out.println(dv[2]);
    
    cube.append(new double[]{0.4, 1, 2, 3, 4, 5, 6, 0.25});
  }
}
