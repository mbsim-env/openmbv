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
    
    group.write(true, true);
    
    String ret=cube.getName(); System.out.println(ret);
    boolean b=cube.getReferenceFrame(); System.out.println(b);
    double d=cube.getLength(); System.out.println(d);
    VectorDouble dv=cuboid.getLength(); System.out.println(dv.get(0)); System.out.println(dv.get(1)); System.out.println(dv.get(2));

    VectorDouble vec=new VectorDouble();
    vec.add(5.6); vec.add(9.7); vec.add(3.5);
    cuboid.setLength(vec);
    dv=cuboid.getLength(); System.out.println(dv.get(0)); System.out.println(dv.get(1)); System.out.println(dv.get(2));
    
    VectorDouble append=new VectorDouble();
    append.add(0.4);
    append.add(1);
    append.add(2);
    append.add(3);
    append.add(4);
    append.add(5);
    append.add(6);
    append.add(0.25);
    cube.append(append);
  }
}
