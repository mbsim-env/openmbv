<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="2.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns:pv="http://www.amm.mw.tu-muenchen.de/XXX/physicalvariable">

  <xsl:output method="xml" version="1.0" indent="yes"/>

  <xsl:import-schema namespace="http://www.amm.mw.tu-muenchen.de/YYY" schema-location="test.xsd"/>

  <!-- clone -->
  <xsl:template match="*">
    <xsl:param name="NUMBER"/>
    <xsl:param name="COUNT"/>
    <xsl:param name="HREF"/>
    <xsl:copy>
      <xsl:if test="$HREF">
        <xsl:attribute name="xml:base">
          <xsl:value-of select="$HREF"/>
        </xsl:attribute>
      </xsl:if>
      <xsl:apply-templates select="@*">
        <xsl:with-param name="NUMBER" select="$NUMBER"/>
        <xsl:with-param name="COUNT" select="$COUNT"/>
      </xsl:apply-templates>
      <xsl:apply-templates>
        <xsl:with-param name="NUMBER" select="$NUMBER"/>
        <xsl:with-param name="COUNT" select="$COUNT"/>
      </xsl:apply-templates>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="@*">
    <xsl:copy/>
  </xsl:template>

  <!-- TODO: fix double inclusion of xsi:schemaLocation of the same namespace !!!!!!!!!!!!!!!!! -->
  <xsl:template match="@xsi:schemaLocation">
    <xsl:if test="not(../@name)">
      <xsl:copy/>
    </xsl:if>
  </xsl:template>



  <!-- embed a source document count times -->
  <xsl:template match="pv:embed">
    <xsl:call-template name="EMBED">
      <xsl:with-param name="NUMBER" select="0"/>
      <xsl:with-param name="COUNT" select="@count"/>
      <xsl:with-param name="HREF" select="@href"/>
    </xsl:call-template>
  </xsl:template>

  <!-- recursive embed the source document -->
  <xsl:template name="EMBED">
   <xsl:param name="NUMBER"/>
   <xsl:param name="COUNT"/>
   <xsl:param name="HREF"/>
   <xsl:if test="$NUMBER &lt; $COUNT">
     <xsl:apply-templates select="document($HREF)">
       <xsl:with-param name="NUMBER" select="$NUMBER"/>
       <xsl:with-param name="COUNT" select="$COUNT"/>
       <xsl:with-param name="HREF" select="$HREF"/>
     </xsl:apply-templates>
     <xsl:call-template name="EMBED">
       <xsl:with-param name="NUMBER" select="$NUMBER+1"/>
       <xsl:with-param name="COUNT" select="$COUNT"/>
       <xsl:with-param name="HREF" select="$HREF"/>
     </xsl:call-template>
   </xsl:if>
  </xsl:template>

  <!-- clone pv:scalar, pv:vector, pv:matrix and resubstitute number and count -->
  <xsl:template match="element(*,pv:scalar)|element(*,pv:vector)|element(*,pv:matrix)">
    <xsl:param name="NUMBER"/>
    <xsl:param name="COUNT"/>
    <xsl:apply-templates mode="PV" select=".">
      <xsl:with-param name="NUMBER" select="$NUMBER"/>
      <xsl:with-param name="COUNT" select="$COUNT"/>
    </xsl:apply-templates>
  </xsl:template>

  <!-- resubstitute number and count in text node -->
  <xsl:template mode="PV" match="*">
    <xsl:param name="NUMBER"/>
    <xsl:param name="COUNT"/>
    <xsl:copy>
      <xsl:apply-templates select="@*"/>
      <xsl:if test="child::*">
        <xsl:apply-templates mode="PV" select="child::*">
          <xsl:with-param name="NUMBER" select="$NUMBER"/>
          <xsl:with-param name="COUNT" select="$COUNT"/>
        </xsl:apply-templates>
      </xsl:if>
      <xsl:if test="not(child::*) and text()">
        <xsl:value-of select="replace(replace(.,'([^a-zA-Z0-9_]|^)number([^a-zA-Z0-9_]|$)',concat('$1(',string($NUMBER),')$2')),'([^a-zA-Z0-9_]|^)count([^a-zA-Z0-9_]|$)',concat('$1(',string($COUNT),')$2'))"/>
      </xsl:if>
    </xsl:copy>
  </xsl:template>
  
  <!-- resubstitute number and count in name attribute -->
  <xsl:template match="@name">
    <xsl:param name="NUMBER"/>
    <xsl:param name="COUNT"/>
    <xsl:attribute name="name"><xsl:value-of select="replace(.,'@number@',string($NUMBER))"/></xsl:attribute>
  </xsl:template>

</xsl:stylesheet>
