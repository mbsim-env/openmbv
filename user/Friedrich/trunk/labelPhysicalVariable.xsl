<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="2.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:ty="http://www.amm.mw.tu-muenchen.de/XXX/physicalvariable">

  <xsl:output method="xml" version="1.0" indent="yes"/>

  <!-- clone -->

  <xsl:template match="*">
    <xsl:copy>
      <xsl:apply-templates select="@*"/>
      <xsl:apply-templates/>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="@*">
    <xsl:copy/>
  </xsl:template>



  <!-- deunit -->

  <xsl:import-schema namespace="@NAMESPACE@" schema-location="@NAMESPACELOCATION@"/>

  <xsl:template match="element(*,ty:scalar)">
    <xsl:copy>@SCALAREXPRESSION{@<xsl:value-of select="normalize-space(.)"/>@}@</xsl:copy>
  </xsl:template>

  <xsl:template match="element(*,ty:vector)">
    <xsl:copy>@VECTOREXPRESSION{@<xsl:value-of select="normalize-space(.)"/>@}@</xsl:copy>
  </xsl:template>

  <xsl:template match="element(*,ty:matrix)">
    <xsl:copy>@MATRIXEXPRESSION{@<xsl:value-of select="normalize-space(.)"/>@}@</xsl:copy>
  </xsl:template>

</xsl:stylesheet>
