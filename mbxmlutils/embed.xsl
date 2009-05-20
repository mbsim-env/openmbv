<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="2.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable">

  <xsl:output method="xml" version="1.0" indent="yes"/>

  <xsl:import-schema namespace="@NAMESPACE@" schema-location="@NAMESPACELOCATION@"/>

  <!-- clone -->
  <xsl:template match="*">
    <xsl:param name="NUMBER"/>
    <xsl:param name="COUNT"/>
    <xsl:param name="COUNTERNAME"/>
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
        <xsl:with-param name="COUNTERNAME" select="$COUNTERNAME"/>
      </xsl:apply-templates>
      <xsl:apply-templates>
        <xsl:with-param name="NUMBER" select="$NUMBER"/>
        <xsl:with-param name="COUNT" select="$COUNT"/>
        <xsl:with-param name="COUNTERNAME" select="$COUNTERNAME"/>
      </xsl:apply-templates>
    </xsl:copy>
  </xsl:template>

  <!-- TODO: fix double inclusion of xsi:schemaLocation of the same namespace !!!!!!!!!!!!!!!!! -->
  <xsl:template match="@xsi:schemaLocation">
    <xsl:if test="name(..)='DynamicSystemSolver'">
      <xsl:copy/>
    </xsl:if>
  </xsl:template>



  <!-- embed a source document count times -->
  <xsl:template match="pv:embed">
    <xsl:call-template name="EMBED">
      <xsl:with-param name="NUMBER" select="1"/>
      <xsl:with-param name="COUNT" select="@count"/>
      <xsl:with-param name="COUNTERNAME" select="@counterName"/>
      <xsl:with-param name="HREF" select="@href"/>
    </xsl:call-template>
  </xsl:template>

  <!-- recursive embed the source document -->
  <xsl:template name="EMBED">
   <xsl:param name="NUMBER"/>
   <xsl:param name="COUNT"/>
   <xsl:param name="COUNTERNAME"/>
   <xsl:param name="HREF"/>
   <xsl:if test="$NUMBER &lt;= $COUNT">
     <xsl:apply-templates select="document($HREF)">
       <xsl:with-param name="NUMBER" select="$NUMBER"/>
       <xsl:with-param name="COUNT" select="$COUNT"/>
       <xsl:with-param name="COUNTERNAME" select="$COUNTERNAME"/>
       <xsl:with-param name="HREF" select="$HREF"/>
     </xsl:apply-templates>
     <xsl:call-template name="EMBED">
       <xsl:with-param name="NUMBER" select="$NUMBER+1"/>
       <xsl:with-param name="COUNT" select="$COUNT"/>
       <xsl:with-param name="COUNTERNAME" select="$COUNTERNAME"/>
       <xsl:with-param name="HREF" select="$HREF"/>
     </xsl:call-template>
   </xsl:if>
  </xsl:template>

  <!-- clone pv:scalar, pv:vector, pv:matrix and resubstitute number and count -->
  <xsl:template match="element(*,pv:scalar)|element(*,pv:vector)|element(*,pv:matrix)">
    <xsl:param name="NUMBER"/>
    <xsl:param name="COUNT"/>
    <xsl:param name="COUNTERNAME"/>
    <xsl:apply-templates mode="PV" select=".">
      <xsl:with-param name="NUMBER" select="$NUMBER"/>
      <xsl:with-param name="COUNT" select="$COUNT"/>
      <xsl:with-param name="COUNTERNAME" select="$COUNTERNAME"/>
    </xsl:apply-templates>
  </xsl:template>

  <!-- resubstitute number and count in text nodes -->
  <xsl:template mode="PV" match="*">
    <xsl:param name="NUMBER"/>
    <xsl:param name="COUNT"/>
    <xsl:param name="COUNTERNAME"/>
    <xsl:copy>
      <xsl:apply-templates select="@*">
        <xsl:with-param name="NUMBER" select="$NUMBER"/>
        <xsl:with-param name="COUNT" select="$COUNT"/>
        <xsl:with-param name="COUNTERNAME" select="$COUNTERNAME"/>
      </xsl:apply-templates>
      <xsl:if test="child::*">
        <xsl:apply-templates mode="PV" select="child::*">
          <xsl:with-param name="NUMBER" select="$NUMBER"/>
          <xsl:with-param name="COUNT" select="$COUNT"/>
          <xsl:with-param name="COUNTERNAME" select="$COUNTERNAME"/>
        </xsl:apply-templates>
      </xsl:if>
      <xsl:if test="not(child::*) and text() and $COUNTERNAME!=''">
        <xsl:value-of select="replace(replace(.,concat('([^a-zA-Z0-9_]|^)',$COUNTERNAME,'([^a-zA-Z0-9_]|$)'),concat('$1(',$NUMBER,')$2')),concat('([^a-zA-Z0-9_]|^)',$COUNTERNAME,'MAX','([^a-zA-Z0-9_]|$)'),concat('$1(',$COUNT,')$2'))"/>
      </xsl:if>
      <xsl:if test="not(child::*) and text() and $COUNTERNAME=''">
        <xsl:value-of select="."/>
      </xsl:if>
    </xsl:copy>
  </xsl:template>
  
  <!-- resubstitute number and count in attributes -->
  <xsl:template match="@*">
    <xsl:param name="NUMBER"/>
    <xsl:param name="COUNT"/>
    <xsl:param name="COUNTERNAME"/>
    <xsl:attribute name="{name()}">
      <xsl:if test="$COUNTERNAME!='' and (name()='name' or matches(name(),'^ref'))">
        <xsl:value-of select="replace(replace(.,concat('([^a-zA-Z0-9_]|^)',$COUNTERNAME,'([^a-zA-Z0-9_]|$)'),concat('$1(',$NUMBER,')$2')),concat('([^a-zA-Z0-9_]|^)',$COUNTERNAME,'MAX','([^a-zA-Z0-9_]|$)'),concat('$1(',$COUNT,')$2'))"/>
      </xsl:if>
      <xsl:if test="$COUNTERNAME='' or (name()!='name' and not(matches(name(),'^ref')))">
        <xsl:value-of select="."/>
      </xsl:if>
    </xsl:attribute>
  </xsl:template>

</xsl:stylesheet>
