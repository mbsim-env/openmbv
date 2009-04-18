<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:av="http://www.amm.mw.tum.de/AMVis">

  <xsl:output method="text"/>

  <xsl:template match="/av:Group">
    <xsl:apply-templates select="*">
      <xsl:with-param name="FULLNAME" select="''"/>
    </xsl:apply-templates>
  </xsl:template>

  <xsl:template match="av:Group">
    <xsl:param name="FULLNAME"/>
    <xsl:apply-templates select="*">
      <xsl:with-param name="FULLNAME" select="concat($FULLNAME,'.',@name)"/>
    </xsl:apply-templates>
  </xsl:template>

  <xsl:template match="av:Cuboid">
    <xsl:param name="FULLNAME"/>FILENAME: <xsl:value-of select="concat($FULLNAME,'.',@name)"/>.data
Cuboid
1
0
<xsl:value-of select="av:initialTranslation"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:initialRotation"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:scaleFactor"/><xsl:text>
</xsl:text>
<xsl:value-of select="concat('{',av:length,'}')"/><xsl:text>
</xsl:text></xsl:template>

  <xsl:template match="av:Frame">
    <xsl:param name="FULLNAME"/>FILENAME: <xsl:value-of select="concat($FULLNAME,'.',@name)"/>.data
Kos
1
0
<xsl:value-of select="av:initialTranslation"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:initialRotation"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:scaleFactor"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:size"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:offset"/><xsl:text>
</xsl:text></xsl:template>

  <xsl:template match="av:Arrow">
    <xsl:param name="FULLNAME"/>FILENAME: <xsl:value-of select="concat($FULLNAME,'.',@name)"/>.data
Arrow
1
0
<xsl:value-of select="av:diameter"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:headDiameter"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:headLength"/><xsl:text>
</xsl:text>1<xsl:text>
</xsl:text>
  </xsl:template>

  <xsl:template match="av:Cube">
    <xsl:param name="FULLNAME"/>FILENAME: <xsl:value-of select="concat($FULLNAME,'.',@name)"/>.data
Cube
1
0
<xsl:value-of select="av:initialTranslation"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:initialRotation"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:scaleFactor"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:length"/><xsl:text>
</xsl:text>
  </xsl:template>

  <xsl:template match="av:Cylinder">
    <xsl:param name="FULLNAME"/>FILENAME: <xsl:value-of select="concat($FULLNAME,'.',@name)"/>.data
Cylinder
1
0
<xsl:value-of select="av:initialTranslation"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:initialRotation"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:scaleFactor"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:baseRadius"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:topRadius"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:height"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:innerBaseRadius"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:innerTopRadius"/><xsl:text>
</xsl:text>
  </xsl:template>

  <xsl:template match="av:Extrusion">
    <xsl:param name="FULLNAME"/>FILENAME: <xsl:value-of select="concat($FULLNAME,'.',@name)"/>.data
Extrusion
1
0
<xsl:value-of select="av:initialTranslation"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:initialRotation"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:scaleFactor"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:windingRule"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:height"/><xsl:text>
</xsl:text><xsl:apply-templates select="av:contour"/>0 0 -1
</xsl:template>
  <xsl:template match="av:contour"><xsl:value-of select="."/>0 0 -2
</xsl:template>

  <xsl:template match="av:Rotation">
    <xsl:param name="FULLNAME"/>FILENAME: <xsl:value-of select="concat($FULLNAME,'.',@name)"/>.data
Rotation
1
0
<xsl:value-of select="av:initialTranslation"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:initialRotation"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:scaleFactor"/><xsl:text>
</xsl:text><xsl:apply-templates select="av:contour"/>0 0 -1
</xsl:template>

  <xsl:template match="av:Sphere">
    <xsl:param name="FULLNAME"/>FILENAME: <xsl:value-of select="concat($FULLNAME,'.',@name)"/>.data
Sphere
1
0
<xsl:value-of select="av:initialTranslation"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:initialRotation"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:scaleFactor"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:radius"/><xsl:text>
</xsl:text>
  </xsl:template>

  <xsl:template match="av:CoilSpring">
    <xsl:param name="FULLNAME"/>FILENAME: <xsl:value-of select="concat($FULLNAME,'.',@name)"/>.data
CoilSpring
1
0
<xsl:value-of select="av:numberOfCoils"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:springRadius"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:crossSectionRadius"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:scaleFactor"/><xsl:text>
</xsl:text>
  </xsl:template>

  <xsl:template match="av:ObjObject">
    <xsl:param name="FULLNAME"/>FILENAME: <xsl:value-of select="concat($FULLNAME,'.',@name)"/>.data
ObjObject
1
0
<xsl:value-of select="av:initialTranslation"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:initialRotation"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:scaleFactor"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:objFileName"/>
1
1
0
<xsl:value-of select="av:epsVertex"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:epsNormal"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:smoothBarrier"/>
0
</xsl:template>

  <xsl:template match="av:InvisibleBody">
    <xsl:param name="FULLNAME"/>FILENAME: <xsl:value-of select="concat($FULLNAME,'.',@name)"/>.data
InvisibleBody
1
0
<xsl:value-of select="av:initialTranslation"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:initialRotation"/><xsl:text>
</xsl:text>
<xsl:value-of select="av:scaleFactor"/><xsl:text>
</xsl:text>
</xsl:template>

  <xsl:template match="av:Path">
    <xsl:param name="FULLNAME"/>FILENAME: <xsl:value-of select="concat($FULLNAME,'.',@name)"/>.data
Path
1
0
<xsl:value-of select="av:color"/><xsl:text>
</xsl:text>
</xsl:template>

</xsl:stylesheet>
