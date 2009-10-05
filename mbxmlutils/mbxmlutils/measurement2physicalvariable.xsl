<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:mm="http://openmbv.berlios.de/MBXMLUtils/measurement"
  xmlns:xs="http://www.w3.org/2001/XMLSchema">

  <xsl:param name="SCHEMADIR"/>

  <xsl:output method="xml" version="1.0" indent="yes"/>

  <xsl:template match="text()"/>

  <xsl:template match="/mm:measurement">
  
    <xsl:comment>
      <!-- Add this text to the output document to prevent others editing the output document manual -->
      This file is generated automatically using XSLT from measurement.xml.
      DO NOT EDIT!!!
    </xsl:comment>

    <xs:schema targetNamespace="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"
      elementFormDefault="qualified"
      attributeFormDefault="unqualified"
      xmlns="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"
      xmlns:xml="http://www.w3.org/XML/1998/namespace"
      xmlns:p="http://openmbv.berlios.de/MBXMLUtils/parameter"
      xmlns:xs="http://www.w3.org/2001/XMLSchema">

      <!-- for xml:base attribute added by XInclude aware parser: include xml namespaces defining attribute xml:base -->
      <xs:import namespace="http://www.w3.org/XML/1998/namespace" schemaLocation="{$SCHEMADIR}/xml.xsd"/>

      <xs:import namespace="http://openmbv.berlios.de/MBXMLUtils/parameter" schemaLocation="{$SCHEMADIR}/parameter.xsd"/>

      <!-- element for embeding -->
      <xs:element name="embed">
        <xs:complexType>
          <xs:sequence>
            <xs:element name="localParameter" minOccurs="0">
              <xs:complexType>
                <xs:sequence>
                  <xs:element ref="p:parameter"/>
                </xs:sequence>
              </xs:complexType>
            </xs:element>
            <xs:any namespace="##other" processContents="strict" minOccurs="0"/>
          </xs:sequence>
          <xs:attribute name="href" type="xs:anyURI" use="optional"/>
          <xs:attribute name="count" use="optional" type="fullOctaveString" default="1"/>
          <xs:attribute name="counterName" use="optional">
            <xs:simpleType>
              <xs:restriction base="xs:token">
                <xs:pattern value="[a-zA-Z_][a-zA-Z0-9_]*"/>
              </xs:restriction>
            </xs:simpleType>
          </xs:attribute>
          <xs:attribute name="onlyif" use="optional" type="fullOctaveString" default="1"/>
        </xs:complexType>
      </xs:element>

      <!-- base type for a string which is fully converted by octave -->
      <xs:simpleType name="fullOctaveString">
        <xs:restriction base="xs:token"/>
      </xs:simpleType>

      <!-- base type for a string which is partially converted by octave.
           Only the content between { and } ist converted by octave -->
      <xs:simpleType name="partialOctaveString">
        <xs:restriction base="xs:token"/>
      </xs:simpleType>

      <!-- A regexp for matching a MBXMLUtils name attribute. E.g. matches: "box1", "box{i+5}", "box2_{2*i+1}_{j+1}" -->
      <xs:simpleType name="name">
        <xs:restriction base="partialOctaveString">
          <xs:pattern>
            <xsl:attribute name="value">((([a-zA-Z_]|[a-zA-Z_][a-zA-Z0-9_]*[a-zA-Z_])\{[^\}]+\})+([a-zA-Z_][a-zA-Z0-9_]*)?|[a-zA-Z_][a-zA-Z0-9_]*)</xsl:attribute>
          </xs:pattern>
        </xs:restriction>
      </xs:simpleType>

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
        </xs:choice>
      </xs:complexType>

      <!-- add scalar units -->
      <xsl:apply-templates mode="SCALAR" select="/mm:measurement/mm:measure"/>

      <!-- unknown scalar/vector/matrix -->
      <xs:complexType name="unknownScalar" mixed="true">
        <xs:annotation>
          <xs:documentation>
            A scalar value in a unknown unit. The value is evaluated by the
            octave string given in the convertUnit attribute.
          </xs:documentation>
        </xs:annotation>
        <xs:complexContent>
          <xs:extension base="scalar">
            <xs:attribute name="convertUnit" type="fullOctaveString"/>
          </xs:extension>
        </xs:complexContent>
      </xs:complexType>

      <xs:complexType name="unknownVector" mixed="true">
        <xs:annotation>
          <xs:documentation>
            A vector value in a unknown unit. The value is evaluated by the
            octave string given in the convertUnit attribute.
          </xs:documentation>
        </xs:annotation>
        <xs:complexContent>
          <xs:extension base="vector">
            <xs:attribute name="convertUnit" type="fullOctaveString"/>
          </xs:extension>
        </xs:complexContent>
      </xs:complexType>

      <xs:complexType name="unknownMatrix" mixed="true">
        <xs:annotation>
          <xs:documentation>
            A matrix value in a unknown unit. The value is evaluated by the
            octave string given in the convertUnit attribute.
          </xs:documentation>
        </xs:annotation>
        <xs:complexContent>
          <xs:extension base="matrix">
            <xs:attribute name="convertUnit" type="fullOctaveString"/>
          </xs:extension>
        </xs:complexContent>
      </xs:complexType>

      <!-- string must be enclosed in '"' (processed by octave) -->
      <xs:simpleType name="string">
        <xs:annotation>
          <xs:documentation>
            A string value. The value is evaluated by the octave, so a plain string
            must be enclosed in '"'.
          </xs:documentation>
        </xs:annotation>
        <xs:restriction base="xs:string"/>
      </xs:simpleType>

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
