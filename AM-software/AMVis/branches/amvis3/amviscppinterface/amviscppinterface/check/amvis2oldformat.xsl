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

</xsl:stylesheet>
