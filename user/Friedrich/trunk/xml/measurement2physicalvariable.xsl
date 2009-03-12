<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="2.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:mm="http://www.amm.mw.tu-muenchen.de/XXX/measurement"
  xmlns:xs="http://www.w3.org/2001/XMLSchema">

  <xsl:output method="xml" version="1.0" indent="yes"/>

  <xsl:template match="text()"/>

  <xsl:template match="/mm:measurement">
  
    <xsl:comment>
      <!-- Add this text to the output document to prevent others editing the output document manual -->
      This file is generated automatically using XSLT from measurement.xml.
      DO NOT EDIT!!!
    </xsl:comment>

    <xs:schema targetNamespace="http://www.amm.mw.tu-muenchen.de/XXX/physicalvariable"
      elementFormDefault="qualified"
      attributeFormDefault="unqualified"
      xmlns="http://www.amm.mw.tu-muenchen.de/XXX/physicalvariable"
      xmlns:xs="http://www.w3.org/2001/XMLSchema">

      <!-- for xml:base attribute added by XInclude aware parser: include xml namespaces defining attribute xml:base -->
      <xs:import namespace="http://www.w3.org/XML/1998/namespace" schemaLocation="xml.xsd"/>

      <!-- ################################ -->
      <xs:element name="embed">
        <xs:complexType>
          <xs:attribute name="href" type="xs:anyURI" use="required"/>
          <xs:attribute name="count" type="xs:integer" use="required"/>
          <xs:attribute name="counterName" type="xs:token" use="required"/>
        </xs:complexType>
      </xs:element>
      <!-- ################################ -->
      <!-- add unit types -->
      <xsl:apply-templates mode="UNIT" select="mm:measure"/>

      <xs:complexType mixed="true" name="matrix">
        <xs:annotation>
          <xs:documentation>
            Definition of a matrix of double values
          </xs:documentation>
          <xs:appinfo>
            <pattern xmlns="http://www.ascc.net/xml/schematron" name="test">
              <rule context="event">
                <assert test="not(node()[2])">
                  Only one child node is allowed: text or element.
                  <!-- regex for matrix expression -->
                </assert>
              </rule>
            </pattern>
          </xs:appinfo>
        </xs:annotation>
        <xs:choice minOccurs="0">
          <xs:element name="xmlMatrix">
            <xs:complexType>
              <xs:sequence>
                <xs:element name="row" maxOccurs="unbounded">
                  <xs:complexType>
                    <xs:sequence>
                      <xs:element name="ele" maxOccurs="unbounded">
                        <xs:simpleType>
                          <xs:restriction base="xs:string">
                            <xs:pattern value="\s*.+\s*"/><!-- TODO: add regex for scalar expression (change '.+')-->
                          </xs:restriction>
                        </xs:simpleType>
                      </xs:element>
                    </xs:sequence>
                  </xs:complexType>
                </xs:element>
              </xs:sequence>
              <xs:attribute ref="xml:base"/> <!-- allow a XInclude here -->
            </xs:complexType>
          </xs:element>
          <xs:element name="asciiMatrixRef">
            <xs:complexType>
              <xs:attribute name="href" type="xs:anyURI" use="required"/>
            </xs:complexType>
          </xs:element>
        </xs:choice>
      </xs:complexType>

      <!-- add matrix units -->
      <xsl:apply-templates mode="MATRIX" select="/mm:measurement/mm:measure"/>

      <xs:complexType mixed="true" name="vector">
        <xs:annotation>
          <xs:documentation>
            Definition of a vector of double values
          </xs:documentation>
          <xs:appinfo>
            <pattern xmlns="http://www.ascc.net/xml/schematron" name="test">
              <rule context="event">
                <assert test="not(node()[2])">
                  Only one child node is allowed: text or element.
                  <!-- regex for vector expression -->
                </assert>
              </rule>
            </pattern>
          </xs:appinfo>
        </xs:annotation>
        <xs:choice minOccurs="0">
          <xs:element name="xmlVector">
            <xs:complexType>
              <xs:sequence>
                <xs:element name="ele" maxOccurs="unbounded">
                  <xs:simpleType>
                    <xs:restriction base="xs:string">
                      <xs:pattern value="\s*.+\s*"/><!-- TODO: add regex for scalar expression (change '.+')-->
                    </xs:restriction>
                  </xs:simpleType>
                </xs:element>
              </xs:sequence>
              <xs:attribute ref="xml:base"/> <!-- allow a XInclude here -->
            </xs:complexType>
          </xs:element>
          <xs:element name="asciiVectorRef">
            <xs:complexType>
              <xs:attribute name="href" type="xs:anyURI" use="required"/>
            </xs:complexType>
          </xs:element>
        </xs:choice>
      </xs:complexType>

      <!-- add vector units -->
      <xsl:apply-templates mode="VECTOR" select="/mm:measurement/mm:measure"/>

      <!-- add scalar type -->
      <xs:complexType mixed="true" name="scalar">
        <xs:annotation>
          <xs:documentation>
            Definition of a scalar of double values
          </xs:documentation>
          <xs:appinfo>
            <pattern xmlns="http://www.ascc.net/xml/schematron" name="test">
              <rule context="event">
                <assert test="not(node()[2])">
                  Only one child node is allowed: text or element.
                  <!-- regex for scalar expression -->
                </assert>
              </rule>
            </pattern>
          </xs:appinfo>
        </xs:annotation>
        <xs:choice minOccurs="0">
          <xs:element name="asciiScalarRef">
            <xs:complexType>
              <xs:attribute name="href" type="xs:anyURI" use="required"/>
            </xs:complexType>
          </xs:element>
        </xs:choice>
      </xs:complexType>

      <!-- add scalar units -->
      <xsl:apply-templates mode="SCALAR" select="/mm:measurement/mm:measure"/>

    </xs:schema>

  </xsl:template>

  <!-- add unit types -->
  <xsl:template mode="UNIT" match="/mm:measurement/mm:measure">
    <xs:attributeGroup>
      <xsl:attribute name="name">
        <xsl:value-of select="@name"/>Measure</xsl:attribute>
      <xs:annotation>
        <xs:documentation>
          Enumeration of known units of <xsl:value-of select="@name"/>
        </xs:documentation>
      </xs:annotation>
      <xs:attribute name="unit">
        <xsl:attribute name="default">
          <xsl:value-of select="@SIunit"/>
        </xsl:attribute>
        <xs:simpleType>
          <xs:restriction base="xs:token">
            <xsl:apply-templates mode="UNIT" select="mm:unit"/>
          </xs:restriction>
        </xs:simpleType>
      </xs:attribute>
    </xs:attributeGroup>
  </xsl:template>

  <!-- add unit types -->
  <xsl:template mode="UNIT" match="/mm:measurement/mm:measure/mm:unit">
    <xs:enumeration>
      <xsl:attribute name="value">
        <xsl:value-of select="@name"/>
      </xsl:attribute>
    </xs:enumeration>
  </xsl:template>

  <!-- add matrix units -->
  <xsl:template mode="MATRIX" match="/mm:measurement/mm:measure">
    <xs:complexType mixed="true">
      <xsl:attribute name="name">
        <xsl:value-of select="@name"/>Matrix</xsl:attribute>
      <xs:annotation>
        <xs:documentation>
          Matrix measure of <xsl:value-of select="@name"/>
        </xs:documentation>
      </xs:annotation>
      <xs:complexContent>
        <xs:extension base="matrix">
          <xs:attributeGroup>
            <xsl:attribute name="ref">
              <xsl:value-of select="@name"/>Measure</xsl:attribute>
          </xs:attributeGroup>
        </xs:extension>
      </xs:complexContent>
    </xs:complexType>
  </xsl:template>

  <!-- add vector units -->
  <xsl:template mode="VECTOR" match="/mm:measurement/mm:measure">
    <xs:complexType mixed="true">
      <xsl:attribute name="name">
        <xsl:value-of select="@name"/>Vector</xsl:attribute>
      <xs:annotation>
        <xs:documentation>
          Vector measure of <xsl:value-of select="@name"/>
        </xs:documentation>
      </xs:annotation>
      <xs:complexContent>
        <xs:extension base="vector">
          <xs:attributeGroup>
            <xsl:attribute name="ref">
              <xsl:value-of select="@name"/>Measure</xsl:attribute>
          </xs:attributeGroup>
        </xs:extension>
      </xs:complexContent>
    </xs:complexType>
  </xsl:template>

  <!-- add scalar units -->
  <xsl:template mode="SCALAR" match="/mm:measurement/mm:measure">
    <xs:complexType mixed="true">
      <xsl:attribute name="name">
        <xsl:value-of select="@name"/>Scalar</xsl:attribute>
      <xs:annotation>
        <xs:documentation>
          Scalar measure of <xsl:value-of select="@name"/>
        </xs:documentation>
      </xs:annotation>
      <xs:complexContent>
        <xs:extension base="scalar">
          <xs:attributeGroup>
            <xsl:attribute name="ref">
              <xsl:value-of select="@name"/>Measure</xsl:attribute>
          </xs:attributeGroup>
        </xs:extension>
      </xs:complexContent>
    </xs:complexType>
  </xsl:template>

</xsl:stylesheet>
