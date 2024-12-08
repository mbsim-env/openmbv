<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:mm="http://www.mbsim-env.de/MBXMLUtils/measurement"
  version="1.0">

  <xsl:param name="DATETIME"/>

  <!-- output method -->
  <xsl:output method="html" encoding="UTF-8"/>

  <!-- no default text -->
  <xsl:template match="text()"/>



  <!-- for all direct root child elements (classes) -->
  <xsl:template match="/">
    <!-- html header -->
    <xsl:text disable-output-escaping='yes'>&lt;!DOCTYPE html>
</xsl:text>
    <html lang="en">
    <head>
      <title>MBXMLUtils - XML Documentation</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css"/>
      <link rel="shortcut icon" href="/mbsim/html/mbsimenv.ico" type="image/x-icon"/>
      <link rel="icon" href="/mbsim/html/mbsimenv.ico" type="image/x-icon"/>
      <!-- Note: all defined class names and function names here start with _ to differentiate them from bootstrap ones -->
      <style>
        ul._content { padding-left:3ex; list-style-type:none; }
        *._attributeNoMargin { font-family:monospace; font-weight:bold; }
        *._type { font-family:monospace; }
        *._element { font-family:monospace; font-weight:bold; }
      </style>

      <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
          extensions: ["tex2jax.js", "TeX/AMSmath.js", "TeX/AMSsymbols.js"],
          jax: ["input/TeX","output/HTML-CSS"],
          tex2jax: {
            inlineMath: [['\\(','\\)']],
            displayMath: [['\\[','\\]']],
          },
        });
      </script>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js"> </script>
    </head>
    <body style="margin:0.5em">
    <script src="https://code.jquery.com/jquery-2.1.4.min.js"> </script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/moment.js/2.22.2/moment.min.js"> </script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/moment-timezone/0.5.23/moment-timezone-with-data-2012-2022.min.js"> </script>
    <script>
      $(document).ready(function() {
        $('.DATETIME').each(function() {
          $(this).text(moment($(this).text()).tz(moment.tz.guess()).format("ddd, YYYY-MM-DD - HH:mm:ss z"));
        }); 
      });
    </script>
    <div class="page-header">
      <h1>MBXMLUtils - XML Documentation</h1>
      <p>XML-Namespace: <span class="label label-warning">http://www.mbsim-env.de/MBXMLUtils</span></p>
    </div>
    <div class="h2">Contents</div>
    <ul class="_content">
      <li><a id="introduction-content" href="#introduction">1 Introduction</a></li>
      <li><a id="legend-content" href="#legend">2 Legend</a></li>
      <li><a id="name-content" href="#name">3 Element Name</a></li>
      <li><a id="type-content" href="#type">4 Types</a>
        <ul class="_content">
          <li><a id="scalartype-content" href="#scalartype">4.1 Scalar Type</a></li>
          <li><a id="vectortype-content" href="#vectortype">4.2 Vector Type</a></li>
          <li><a id="matrixtype-content" href="#matrixtype">4.3 Matrix Type</a></li>
        </ul>
      </li>
      <li><a id="parameters-content" href="#parameters">5 Parameters</a></li>
      <li><a id="evaluator-content" href="#evaluator">6 Expression Evaluator</a>
        <ul class="_content">
          <li><a id="octaveevaluator-content" href="#octaveevaluator">6.1 Octave Expression Evaluator</a></li>
          <li><a id="pythonevaluator-content" href="#pythonevaluator">6.2 Python Expression Evaluator</a></li>
        </ul>
      </li>
      <li><a id="symbolicFunctions-content" href="#symbolicFunctions">7 Symbolic Functions</a>
        <ul class="_content">
          <li><a id="octavesymbolicFunctions-content" href="#octavesymbolicFunctions">7.1 Octave Symbolic Functions</a></li>
          <li><a id="pythonsymbolicFunctions-content" href="#pythonsymbolicFunctions">7.2 Python Symbolic Functions</a></li>
        </ul>
      </li>
      <li><a id="embed-content" href="#embed">8 Embeding</a></li>
      <li><a id="measurements-content" href="#measurements">9 Measurements</a>
        <ul class="_content">
          <xsl:for-each select="/mm:measurement/mm:measure">
            <xsl:sort select="@name"/>
            <li>
              <a><xsl:attribute name="id"><xsl:value-of select="@name"/>-content</xsl:attribute>
                <xsl:attribute name="href">#<xsl:value-of select="@name"/></xsl:attribute>
                <span class="glyphicon glyphicon-unchecked"/><xsl:text> </xsl:text><xsl:value-of select="@name"/></a>
            </li>
          </xsl:for-each>
        </ul>
      </li>
    </ul>
    <hr class="_hr"/>

    <h1><a id="introduction" href="#introduction-content">1 Introduction</a></h1>

    <h1><a id="legend" href="#legend-content">2 Legend</a></h1>
    <table class="table table-condensed">
      <thead>
        <tr><th>Icon</th><th>Description</th></tr>
      </thead>
      <tbody>
        <tr><td><span class="_element">&lt;element&gt;</span></td><td>A XML element of name 'element'</td></tr>
        <tr><td><span class="_attributeNoMargin">attrName</span></td><td>A XML attribute of name 'attrName'</td></tr>
        <tr><td><span class="label label-warning">namespace</span></td><td>A XML namespace of name 'namespace'</td></tr>
        <tr><td><span class="label label-info">type</span></td><td>A XML element or attribute type of name 'type'</td></tr>
      </tbody>
    </table>

    <h1><a id="name" href="#name-content">3 Element Name</a></h1>
    <p>Elements which must be referable must have a name. Mostly this name is given by the attribute <span class="_attributeNoMargin">name</span>. A valid name starts with a letter or a underscore. The following characters can be letters, underscores or digits. The content between '<code>{</code>' and '<code>}</code>' can be any <a href="#evaluator">Expression Evaluator</a> and is substituted by the result of the evaluator (which must be a valid name; normal a integer number).</p>
    <p>The following table shows examples for valid element names (on the left) and the substituted names (on the right), if there exist a scalar (integer) parameter of name <code>n</code> with the value <code>2</code>:</p>
    <table class="table table-condensed table-striped table-hover">
      <thead>
        <tr><th>Element Name</th><th>Substituted Element Name</th></tr>
      </thead>
      <tbody>
        <tr><td><code>Object1</code></td><td><code>Object1</code></td></tr>
        <tr><td><code>myname_3</code></td><td><code>myname_3</code></td></tr>
        <tr><td><code>body{n+6}</code></td><td><code>body8</code></td></tr>
        <tr><td><code>Obj_{n+4}_{if n==2; ret=3; else ret=6; end}</code></td><td><code>Obj_6_3</code></td></tr>
      </tbody>
    </table>

    <h1><a id="type" href="#type-content">4 Types</a></h1>
    <h2><a id="scalartype" href="#scalartype-content">4.1 Scalar Type</a>
      <xsl:for-each select="/mm:measurement/mm:measure">
        <a id="{@name}Scalar"/>
      </xsl:for-each>
    </h2>
    <p>A scalar type can be of any unit defined in <a href="#measurements">measurements</a>. The unit is given by a optional
      attribute of name <span class="_attributeNoMargin">unit</span>.
      The type name of a scalar of measure length is <span class="label label-info _type">pv:lengthScalar</span> and so on.
      Where <code>pv</code> is mapped to the namespace-uri <span class="label label-warning">http://www.mbsim-env.de/MBXMLUtils</span>.</p>
    <p>The content of a scalar type must be a <a href="#evaluator">Expression Evaluator</a>. The following examples are valid, if there exist a scalar paremter <code>a</code> and <code>b</code> in the <a href="#parameters">parameter file</a>:</p>
    <pre>&lt;myScalarElement unit="mm"&gt;4*sin(a)+b&lt;/myScalarElement&gt;</pre>
    <pre>&lt;myScalarElement&gt;[a,2]*[3;b]&lt;/myScalarElement&gt;</pre>
    <p>There is also a special unit of name <code>unknown</code> defined. This unit dose not take the optional <code>unit</code>
      attribute, it takes an optional attribute of name <span class="_attributeNoMargin">convertUnit</span>. The value of this attribute can be a
      <a href="#evaluator">Expression Evaluator</a>
      which must contain a parameter of name <code>value</code>. The given value is then converted by this expression.</p>

    <h2><a id="vectortype" href="#vectortype-content">4.2 Vector Type</a>
      <xsl:for-each select="/mm:measurement/mm:measure">
        <a id="{@name}Vector"/>
      </xsl:for-each>
    </h2>
    <p>A vector type can be of any unit defined in <a href="#measurements">measurements</a>. The unit is given by a optional
      attribute of name <span class="_attributeNoMargin">unit</span>.
      The type name of a vector of measure length is <span class="label label-info _type">pv:lengthVector</span> and so on.
      Where <code>pv</code> is mapped to the namespace-uri <span class="label label-warning">http://www.mbsim-env.de/MBXMLUtils</span>.</p>
    <p>The content of a vector type can be one of the following:</p>
    <ul>
      <li>A <a href="#evaluator">Expression Evaluator</a>. The following examples are valid, if there exist a scalar paremter <code>a</code> and <code>b</code> in the <a href="#parameters">parameter file</a>:
          <pre>&lt;myVectorElement unit="mm"&gt;[1;b;a;7]&lt;/myVectorElement&gt;</pre>
          <pre>&lt;myVectorElement&gt;[a,2;5.6,7]*[3;b]&lt;/myVectorElement&gt;</pre>
       <p>Using the corresponding evaluator command it is also possible to load the data from a external file (for octave):</p>
       <pre>&lt;myMatrixElement&gt;ret=load('mydata.dat')&lt;/myMatrixElement&gt;</pre>
      </li>
      <li>A XML representation of a vector: The following shows a example of such a XML representation.<pre>
&lt;myVectorElement xmlns:pv="http://www.mbsim-env.de/MBXMLUtils"&gt;
  &lt;pv:ele&gt;6.5&lt;/pv:ele&gt;
  &lt;pv:ele&gt;1.5&lt;/pv:ele&gt;
  &lt;pv:ele&gt;7.3&lt;/pv:ele&gt;
&lt;/myVectorElement&gt;
</pre></li>
    </ul>
    <p>For the special unit of name <code>unknown</code> see <a href="#scalartype">Scalar Type</a></p>

    <h2><a id="matrixtype" href="#matrixtype-content">4.3 Matrix Type</a>
      <xsl:for-each select="/mm:measurement/mm:measure">
        <a id="{@name}Matrix"/>
      </xsl:for-each>
    </h2>
    <p>A matrix type can be of any unit defined in <a href="#measurements">measurements</a>. The unit is given by a optional
      attribute of name <span class="_attributeNoMargin">unit</span>.
      The type name of a matrix of measure length is <span class="label label-info _type">pv:lengthMatrix</span> and so on.
      Where <code>pv</code> is mapped to the namespace-uri <span class="label label-warning">http://www.mbsim-env.de/MBXMLUtils</span>.</p>
    <p>The content of a matrix type can be one of the following:</p>
    <ul>
      <li>A <a href="#evaluator">Expression Evaluator</a>. The following examples are valid, if there exist a scalar paremter <code>a</code> and <code>b</code> in the <a href="#parameters">parameter file</a>:
          <pre>&lt;myMatrixElement unit="mm"&gt;[1,b;a,7]&lt;/myMatrixElement&gt;</pre>
          <pre>&lt;myMatrixElement&gt;[a,2;5.6,7]*rand(2,2)&lt;/myMatrixElement&gt;</pre>
       <p>Using the corresponding evaluator command it is also possible to load the data from a external file (for octave):</p>
       <pre>&lt;myMatrixElement&gt;ret=load('mydata.dat')&lt;/myMatrixElement&gt;</pre>
      </li>
      <li>A XML representation of a matrix: The following shows a example of such a XML representation.<pre>
&lt;myMatrixElement xmlns:pv="http://www.mbsim-env.de/MBXMLUtils"&gt;
  &lt;pv:row&gt;
    &lt;pv:ele&gt;6.5&lt;/pv:ele&gt;
    &lt;pv:ele&gt;1.5&lt;/pv:ele&gt;
  &lt;/pv:row&gt;
  &lt;pv:row&gt;
    &lt;pv:ele&gt;6.5&lt;/pv:ele&gt;
    &lt;pv:ele&gt;1.5&lt;/pv:ele&gt;
  &lt;/pv:row&gt;
&lt;/myMatrixElement&gt;
</pre></li>
    </ul>
    <p>For the special unit of name <code>unknown</code> see <a href="#scalartype">Scalar Type</a></p>

    <h1><a id="parameters" href="#parameters-content">5 Parameters</a></h1>
    <p>A example for a parameter file is given below:</p>
<pre>&lt;Parameter xmlns="http://www.mbsim-env.de/MBXMLUtils"&gt;
  &lt;scalarParameter name="N"&gt;9&lt;/scalarParameter&gt;
  &lt;vectorParameter name="a"&gt;[1;2;3]*N&lt;/vectorParameter&gt;
  &lt;scalarParameter name="lO"&gt;0.2*N&lt;/scalarParameter&gt;
  &lt;matrixParameter name="A"&gt;[1,2;3,4]&lt;/matrixParameter&gt;
  &lt;anyAarameter name="p"&gt;{'test', 4, 6.0}&lt;/anyParameter&gt;
  &lt;import&gt;'/home/user/octaveScripts'&lt;/import&gt;
  &lt;import type="someString"&gt;'/home/user/octaveScripts'&lt;/import&gt;
&lt;/Parameter&gt;
</pre>
    <p>The parameter names must be unique. The parameters are added from top to bottom. Parameters may depend on parameters already added. The parameter values can be given as <a href="#evaluator">Expression Evaluator</a>. Hence a parameter below another parameter may reference this value.</p>
    <p>&lt;scalarParameter&gt;, &lt;vectorParameter&gt; and &lt;matrixParameter&gt; define a parameter value of type scalar, vector and matrix, respectively. &lt;anyParameter&gt; defines a parameter value of any type the evaluator can handle, e.g. a cell array or struct for octave. &lt;import&gt; is highly dependent on the evaluator and does not have a 'name' attribute but an optional 'type' attribute which defaults to '': it imports submodules, adds to the search path or something else. See at a specific evaluator for details about what it does for this evaluator.</p>

    <h1><a id="evaluator" href="#evaluator-content">6 Expression Evaluator</a></h1>
    <p>Different expression evaluators can be used. Currently implemented is python and octave as evaluator. This description of this general section is based on the octave expression evaluator, but all other evaluators are similar. See the sub-sections for details.</p>
    <p>A octave expression/program can be arbitary octave code. It can be a single statement or a statement list.</p>

   <p>If it is a single statement, then the value for the XML element is just the value of the evaluated octave statement. The type of this value must match the type of the XML element (scalar, vector or matrix; any can hold any value). The following examples shows valid examples for a single octave statement, if a scalar <a href="#parameters">parameter</a> of name <code>a</code> and <code>b</code> exist:</p>
<pre>4</pre>
<pre>b</pre>
<pre>3+a*8</pre>
<pre>[4;a]*[6,b]</pre>

<p>If the text is a statement list, then the value for the XML element is the value of the variable 'ret' which must be set by the statement list. The type of the variable 'ret' must match the type of the XML element (scalar, vector or matrix). The following examples shows valid examples for a octave statement, if a scalar <a href="#parameters">parameter</a> of name <code>a</code> and <code>b</code> exist:</p>
<pre>
if 1==a
  ret=4
else
  ret=8
end
</pre>
<p>results in 4 if a==1 else results in 8.</p>
<pre>
myvar=[1;a]
myvar2=
ret=myvar2*b
dummy=3
</pre>
<p>results in a vector of two elements where the first is 2*b and the second is 2*a*b.</p>

<p>Note that the characters '&amp;', '&lt;' and '&gt;' are not allowed in XML. So you have to quote them with '&amp;amp;', '&amp;lt;' and '&amp;gt;', or you must enclose the octave code in a CDATA section:</p>
<pre>&lt;![CDATA[
function r=myfunc(a)
  r=2*a;
end
if 1 &amp; 2, x=9; else x=8; end
ret=myfunc(m1/2);
]]&gt;
</pre>

<p>The following m-functions extent the octave functionality being useful for several applications:</p>
<dl class="dl-horizontal">
  <dt>\(\text{rotateAboutX}(\varphi)\)</dt>
  <dd>Returns the transformation matrix by angle \(\varphi\); around the x-axis:
    \[
      \left[\begin{array}{ccc}
        1 &amp; 0             &amp; 0              \\
        0 &amp; \cos(\varphi) &amp; -\sin(\varphi) \\
        0 &amp; \sin(\varphi) &amp; \cos(\varphi)
      \end{array}\right]
    \]
  </dd>
  <dt>\(\text{rotateAboutY}(\varphi)\)</dt>
  <dd>Returns the transformation matrix by angle \(\varphi\); around the y-axis:
    \[
      \left[\begin{array}{ccc}
        \cos(\varphi)  &amp; 0 &amp; \sin(\varphi) \\
        0              &amp; 1 &amp; 0             \\
        -\sin(\varphi) &amp; 0 &amp; \cos(\varphi)
      \end{array}\right]
    \]
  </dd>
  <dt>\(\text{rotateAboutZ}(\varphi)\)</dt>
  <dd>Returns the transformation matrix by angle \(\varphi\); around the z-axis:
    \[
      \left[\begin{array}{ccc}
        \cos(\varphi) &amp; -\sin(\varphi) &amp; 0 \\
        \sin(\varphi) &amp; \cos(\varphi)  &amp; 0 \\
        0             &amp; 0              &amp; 1
      \end{array}\right]
    \]
  </dd>
  <dt>\(\text{cardan}(\alpha,\beta,\gamma)\)</dt>
  <dd>Returns the cardan transformation matrix:
    \[
      \left[\begin{array}{ccc}
        \cos(\beta)\cos(\gamma)                                      &amp; -\cos(\beta)\sin(\gamma)                                     &amp; \sin(\beta)              \\
        \cos(\alpha)\sin(\gamma)+\sin(\alpha)\sin(\beta)\cos(\gamma) &amp; \cos(\alpha)\cos(\gamma)-\sin(\alpha)\sin(\beta)\sin(\gamma) &amp; -\sin(\alpha)\cos(\beta) \\
        \sin(\alpha)\sin(\gamma)-\cos(\alpha)\sin(\beta)\cos(\gamma) &amp; \cos(\alpha)\sin(\beta)\sin(\gamma)+\sin(\alpha)\cos(\gamma) &amp; \cos(\alpha)\cos(\beta)
      \end{array}\right]
    \]
  </dd>
  <dt>\(\text{euler}(\Phi,\theta,\varphi)\)</dt>
  <dd>Returns the Euler transformation matrix:
    \[
      \left[\begin{array}{ccc}
        \cos(\varphi)\cos(\Phi)-\sin(\varphi)\cos(\theta)\sin(\Phi) &amp; -\cos(\varphi)\cos(\theta)\sin(\Phi)-\sin(\varphi)\cos(\Phi) &amp; \sin(\theta)\sin(\Phi)  \\
        \cos(\varphi)\sin(\Phi)+\sin(\varphi)\cos(\theta)\cos(\Phi) &amp; \cos(\varphi)\cos(\theta)\cos(\Phi)-\sin(\varphi)\sin(\Phi)  &amp; -\sin(\theta)\cos(\Phi) \\
        \sin(\varphi)\sin(\theta)                                   &amp; \cos(\varphi)\sin(\theta)                                    &amp; \cos(\theta)
      \end{array}\right]
    \]
  </dd>
  <dt>\(\text{installPrefix}()\)</dt>
  <dd>Returns the absolute path of the MBSim-Env installation.</dd>
  <dt>\(\text{invCardan}(T)\)</dt>
  <dd>Returns the inverse of cardan, see above.</dd>
  <dt>\(\text{rgbColor}(r,g,b)\)</dt>
  <dd>Returns the HSV color used by OpenMBV from the given red, green and blue values.</dd>
  <dt>\(\text{tilde}(x)\)</dt>
  <dd>If x is a 3 vector returns the screw symmetric 3x3 matrix for cross product calcuation. If x is a 3x3 matrix the inverse function is used.</dd>
  <dt>\(\text{registerPath}(filename)\)</dt>
  <dd>Adds filename to the list of file on which the model depends (e.g. used for FMU export).</dd>
  <dt>\(\text{getOriginalFilename}()\)</dt>
  <dd>Returns the filename where this statement was defined before the Embeding was done.</dd>
  <dt>\(\text{load}(filename)\)</dt>
  <dd>Loads filename as a CSV file and return the date as a matrix. Details depend on the evaluator, octave or python.</dd>
</dl>

    <h2><a id="octaveevaluator" href="#octaveevaluator-content">6.1 Octave Expression Evaluator</a></h2>
<p>Each evaluation is done in a clean octave context with:</p>
<dl class="dl-horizontal">
  <dt>octave globals:</dt>
  <dd>being preserved between each call but only inside of one instance of the evaluator.</dd>
  <dt>octave locals:</dt>
  <dd>set to the "current parameters"</dd>
</dl>
<p>Where "current parameters" are all parameters of the current parameter stack. The result is the value of the variable
"ret", if given. Else its the value of the variable "ans", if given (note that octave stores the result of expressions
in a variable named "ans"). Else, if the code was just the name of another variable the result is the value of this variable.</p>
<p><code>addImport</code>/<code>&lt;import&gt;</code> does a usual evaluation and the resulting string is added to the octave path [is does more but the exact behaviour is currently not very well defined, try to avoid 'import' in octave].</p>

    <h2><a id="pythonevaluator" href="#pythonevaluator-content">6.2 Python Expression Evaluator</a></h2>
<p>Each evaluation is done in a clean python context with:</p>
<dl class="dl-horizontal">
  <dt>globals():</dt>
  <dd>set to the following key-value pairs, in order, where keys are overwritten if already existing:
    <ul>
      <li>python __builtins__</li>
      <li style="color:gray">the key-values of the global import list (this list is deprecated and should be empty)</li>
      <li>the key-values of the current parameter stack</li>
    </ul>
  </dd>
  <dt>locals():</dt>
  <dd>is the same dictionary as globals()</dd>
</dl>

<p style="color:gray">The deprecated global import list is a dictionary with the same lifetime as the instance of this class, initially empty, and filled with all
new variables defined by each evaluation of <code>addImport</code>/<code>&lt;import&gt;</code> with a type of '' or 'global'. Use it with care (its use is deprecated) to ensure a clear scope of imports.<br/>
All new variables means here all variables which are available in Python globals/locals after the evaluation of the 'import' statement but where not available in Python globals/locals before the evaluation. Hence, if a import add a variable which already existed before it is NOT overwritten. This is different to parameters and a local import which overwrite existing variables with the same name. Overwriting names should be avoid anyway, if possible, to improve readability.</p>

<p>The current parameter stack is a dictionary with a local scope (lifetime). A stack of current parameters exists. Each parameter fills its parameter name to this dictionary overwriting a already existing key. Each <code>addImport</code>/<code>&lt;import&gt;</code> with a type 'local' fills all keys which exist after the evaluation in the python global/local context to this list.</p>

<p>Parameter can be a single expressions where the result of the evaluation is used for the value.
If the evaluation is a multi statement code, then the result is the value of the variable named "ret" which must
be defined by the evaluation. A import is always a multi statement code.</p>

<p>Examples:</p>

<pre>&lt;scalarParameter name="p1"&gt;4+3&lt;/scalarParameter&gt;</pre>
<p>Stores the key "p1" with a value of 7 to the current parameter stack. p1 is assigned the single expression value 4+3.</p>

<pre>&lt;scalarParameter name="p2"&gt;
  a=2
  ret=a+1
&lt;/scalarParameter&gt;</pre>
<p>Stores the key "p2" with a value of 3 to the current parameter stack. p2 is assigned the value of the variable "ret".</p>

<pre>&lt;import type="local"&gt;import numpy&lt;/import&gt;</pre>
<p>Stores the key "numpy" with a struct containing the python numpy module to the current parameter stack. "import numpy" creates a single python variable "numpy" which is stored in the parameter stack.</p>

<pre>&lt;anyParameter name="numpy"&gt;import numpy as ret&lt;/anyParameter&gt;</pre>
<p>Stores the key "numpy" with a struct containing the python numpy module to the current parameter stack. "import numpy as ret" import the python numpy module as the name "ret" which is used by a multi statement code as the value of the parameter name "numpy"</p>

<pre>&lt;import type="local"&gt;from json import *&lt;/import&gt;</pre>
<p>Stores all objects provided by the python module "json" with its name in to the current parameter stack. "from json import *" creates for each object in side of json a variable name which is stored in the parameter stack.</p>

<pre>&lt;import type="local"&gt;
  aa=9
  bb=aa+3
  cc=7
&lt;/import&gt;</pre>
<p>Stores the keys "aa", "bb" and "cc" with the values 9, 12 and 8 to the current parameter stack. All created variables are stored in the parameter stack.</p>

<p>Note than you can store functions (function pointers) using import and also using anyParameter but such functions cannot access python global variables defined using other parameters since python globals() is defined at function define time and keeps the same regardless of where the function is called. See <a href="https://docs.python.org/3/library/functions.html#globals">Python documentation</a>. Hence, pass all required data for such functions via the function parameter list.</p>

    <h1><a id="symbolicFunctions" href="#symbolicFunctions-content">7 Symbolic Functions</a></h1>
    <p>The expression evaluators (octave and python) also support symbolic functions known by the symbolic framework of fmatvec.
    The following table lists all known fmatvec operators and functions of its symbolic framework and its
    corresponding operators and functions in octave and python. In octave these are impelmented
    using SWIG and in python the sympy package is used. Some of the operators and functions can also operate on vector arguments.</p>
    <table class="table table-condensed table-striped table-hover">
      <thead>
        <tr> <th>fmatvec Operator/Function</th> <th>Octave Operator/Function</th> <th>Python Operator/Function</th> </tr>
      </thead>
      <tbody>
        <tr><td><code>a+b</code></td> <td><code>a+b</code></td> <td><code>a+b</code></td></tr>
        <tr><td><code>a-b</code></td> <td><code>a-b</code></td> <td><code>a-b</code></td></tr>
        <tr><td><code>a*b</code></td> <td><code>a*b</code></td> <td><code>a*b</code></td></tr>
        <tr><td><code>a/b</code></td> <td><code>a/b</code></td> <td><code>a/b</code></td></tr>
        <tr><td><code>pow(a,b)</code></td> <td><code>power(a,b)</code></td> <td><code>sympy.Pow(a,b)</code></td></tr>
        <tr><td><code>log(x)</code></td> <td><code>log(x)</code></td> <td><code>sympy.log(x)</code></td></tr>
        <tr><td><code>sqrt(x)</code></td> <td><code>sqrt(x)</code></td> <td><code>sympy.sqrt(x)</code></td></tr>
        <tr><td><code>-x</code></td> <td><code>-x</code></td> <td><code>-x</code></td></tr>
        <tr><td><code>sin(x)</code></td> <td><code>sin(x)</code></td> <td><code>sympy.sin(x)</code></td></tr>
        <tr><td><code>cos(x)</code></td> <td><code>cos(x)</code></td> <td><code>sympy.cos(x)</code></td></tr>
        <tr><td><code>tan(x)</code></td> <td><code>tan(x)</code></td> <td><code>sympy.tan(x)</code></td></tr>
        <tr><td><code>sinh(x)</code></td> <td><code>sinh(x)</code></td> <td><code>sympy.sinh(x)</code></td></tr>
        <tr><td><code>cosh(x)</code></td> <td><code>cosh(x)</code></td> <td><code>sympy.cosh(x)</code></td></tr>
        <tr><td><code>tanh(x)</code></td> <td><code>tanh(x)</code></td> <td><code>sympy.tanh(x)</code></td></tr>
        <tr><td><code>asin(x)</code></td> <td><code>asin(x)</code></td> <td><code>sympy.asin(x)</code></td></tr>
        <tr><td><code>acos(x)</code></td> <td><code>acos(x)</code></td> <td><code>sympy.acos(x)</code></td></tr>
        <tr><td><code>atan(x)</code></td> <td><code>atan(x)</code></td> <td><code>sympy.atan(x)</code></td></tr>
        <tr><td><code>atan2(y,x)</code></td> <td><code>atan2(y,x)</code></td> <td><code>sympy.atan2(y,x)</code></td></tr>
        <tr><td><code>asinh(x)</code></td> <td><code>asinh(x)</code></td> <td><code>sympy.asinh(x)</code></td></tr>
        <tr><td><code>acosh(x)</code></td> <td><code>acosh(x)</code></td> <td><code>sympy.acosh(x)</code></td></tr>
        <tr><td><code>atanh(x)</code></td> <td><code>atanh(x)</code></td> <td><code>sympy.atanh(x)</code></td></tr>
        <tr><td><code>exp(x)</code></td> <td><code>exp(x)</code></td> <td><code>sympy.exp(x)</code></td></tr>
        <tr><td><code>sign(x)</code></td> <td><code>sign(x)</code></td> <td><code>sympy.sign(x)</code></td></tr>
        <tr><td><code>heaviside(x)</code></td> <td><code>heaviside(x)</code></td> <td><code>sympy.Heaviside(x)</code></td></tr>
        <tr><td><code>abs(x)</code></td> <td><code>abs(x)</code></td> <td><code>sympy.Abs(x)</code></td></tr>
        <tr><td><code>min(a,b)</code></td> <td><code>min([a;b])</code> <a href="#fn03">[3]</a></td> <td><code>sympy.Min(a,b)</code> <a href="#fn01">[1]</a></td></tr>
        <tr><td><code>max(a,b)</code></td> <td><code>max([a;b])</code> <a href="#fn03">[3]</a></td> <td><code>sympy.Max(a,b)</code> <a href="#fn01">[1]</a></td></tr>
        <tr><td><code>condition(c,gt,le)</code></td> <td><code>condition(c,gt,le)</code></td> <td><code>sympy.Piecewise((gt, c>0), (le, True))</code> <a href="#fn01">[1]</a> <a href="#fn02">[2]</a></td></tr>
      </tbody>
    </table>
    <footer>
      <ul class="list-group">
        <li id="fn01">[1] more than two arguments are supported. If so, the function is converted to a corresponding nested set of functions with proper number of arguments.</li>
        <li id="fn02">[2] the condition (<code>c&gt;0</code>) can be <code>True</code>, <code>False</code>, <code>&gt;</code>, <code>&gt;=</code>, <code>&lt;</code> or <code>&lt;=</code>. The condition is then converted accordingly. If none of the conditions evaluates to True, 0 is used (python uses None in this case).</li>
        <li id="fn03">[3] the argument is a vector or a scalar. A vector argument is converted to a corresponding nested set of min/max functions with two arguments.</li>
      </ul>
    </footer>

    <h2><a id="octavesymbolicFunctions" href="#octavesymbolicFunctions-content">7.1 Octave Symbolic Functions</a></h2>
<p>As noted above, symbolic functions are implemented in octave using SWIG. Hence, a symbolic scalar is a octave SWIG object.
Symbolic vectors and matrices are usual octave vector and matrices with a element data type of octave SWIG object.</p>

    <h2><a id="pythonsymbolicFunctions" href="#pythonsymbolicFunctions-content">7.2 Python Symbolic Functions</a></h2>
<p>As noted above, symbolic functions are used in the python evaluator using the python sympy module. The full power of sympy
can be used, however, the resulting symbolic expression passed back to the evaluator can only contain a limited set of symbolic
functions. See the above table.</p>
<p>Since the none symbolic part of the python evaluator is based on numpy for vector and matrix representation also symbolic vectors
and matrices are based on numpy: a symblic input vector/matrix to the evaluator is a 1D/2D numpy array of dtype=object (each element
contains a scalar sympy expression).
A vector/matrix output of the evaluator can, however, be of type python list, numpy array or sympy matrix for convenience.
Please note that the helper functions in the mbxmlutils module also output vector/matrix data as a numpy array: either of dtype=float
for pure numeric data or of dtype=object (with scalar sympy expressions) for mixed or pure symbolic data.</p>

    <h1><a id="embed" href="#embed-content">8 Embeding</a></h1>
    <p>Using the <span class="_element">&lt;pv:Embed&gt;</span> element, where the prefix <code>pv</code> is mapped to the namespace-uri <span class="label label-warning">http://www.mbsim-env.de/MBXMLUtils</span> it is possible to embed a XML element multiple times. The full valid example syntax for this element is:</p>
<pre>&lt;pv:Embed href="file.xml" count="2+a" counterName="n" onlyif="n!=2"/&gt;</pre>
<p>or</p>
<pre>&lt;pv:Embed count="2+a" counterName="n" onlyif="n!=2"&gt;
  &lt;any_element_with_childs/&gt;
&lt;/pv:Embed&gt;
</pre>
<p>This will substitute the <span class="_element">&lt;pv:Embed&gt;</span> element in the current context <code>2+a</code> times with the element defined in the file <code>file.xml</code> or with <span class="_element">&lt;any_element_with_childs&gt;</span>. The insert elements have access to the global <a href="#parameters">parameters</a>, see the next paragraph how to extend these parameters. Moreover, the insert elements have access to a parameter named <code>n</code> which counts from <code>1</code> to <code>2+a</code> (or from <code>0</code> to <code>2+a-1</code> for 0-based evaluators) and to a parameter named <code>n_count</code> with the value of the count attibute (2+a).
The new element is only insert if the octave expression defined by the attribute <span class="_attributeNoMargin">onlyif</span> evaluates to <code>1</code> (<code>true</code>). If the attribute <span class="_attributeNoMargin">onlyif</span> is not given it is allways <code>1</code> (<code>true</code>).<br/>
The attributes <span class="_attribure">count</span> and <span class="_attribure">counterName</span> must be given both or none of them. If none are given, then <code>count</code> is <code>1</code> and <span class="_attributeNoMargin">counterName</span> is not used.</p>
<p><span class="_attribure">counterName</span> cannot depend on any parameters.
<span class="_attribure">count</span> can depend on any parameter defined on a higher level.
<span class="_attribure">onlyif</span> can depend on any parameter defined on a higher level and on the couterName.
All parameters and all elements of the embedded content can depend on any parameter defined on a higher level and on the counterName.
<span class="_attribure">href</span> and <span class="_attribure">parameterHref</span> can depend on any parameter define on a higher level but not on counterName.</p>

<p>The first child element of <span class="_element">&lt;pv:Embed&gt;</span> can be the element <span class="_element">&lt;pv:localParameter&gt;</span> which has one child element <a class="_element" href="#parameters">&lt;p:parameter&gt;</a> OR a attribute named <span class="_attributeNoMargin">href</span>. In this case the global parameters are expanded by the parameters given by the element <span class="_element">&lt;p:parameter&gt;</span> or in the file given by <code>href</code>. If a parameter already exist then the parameter is overwritten.</p>
<pre>&lt;pv:Embed count="2+a" counterName="n" onlyif="n!=2"&gt;
  &lt;p:Parameter xmlns:p="http://www.mbsim-env.de/MBXMLUtils"&gt;
    &lt;p:scalarParameter name="h1"&gt;0.5&lt;/p:scalarParameter&gt;
    &lt;p:scalarParameter name="h2"&gt;h1&lt;/p:scalarParameter&gt;
  &lt;/p:Parameter&gt;
  &lt;any_element_with_childs/&gt;
&lt;/pv:Embed&gt;
</pre>

    <h1><a id="measurements" href="#measurements-content">9 Measurements</a></h1>
    <p>The following subsections show all defined measurements.</p>
    <p>The column "Unit Name" in the tables is the name of the unit and the column
      "Conversion to SI Unit" is a expression which converts a value of this unit to the SI unit.</p>
    <p>The <span class="bg-success">highlighted row</span> shows the SI unit of this measurement which always has a conversion of "value".</p>
    <xsl:apply-templates select="/mm:measurement/mm:measure">
      <xsl:sort select="@name"/>
    </xsl:apply-templates>

    <hr/>
    <span class="pull-right small">
      Generated with MBXMLUtils on <span class="DATETIME"><xsl:value-of select="$DATETIME"/></span>
    </span>
    </body></html>
  </xsl:template>

  <xsl:template match="/mm:measurement/mm:measure">
    <h2><a>
      <xsl:attribute name="id">
        <xsl:value-of select="@name"/>
      </xsl:attribute>
      <xsl:attribute name="href">#<xsl:value-of select="@name"/>-content</xsl:attribute>
      <code><xsl:value-of select="@name"/></code>
    </a></h2>
    <table class="table table-condensed table-striped table-hover">
      <thead>
        <tr> <th>Unit Name</th> <th>Conversion to SI Unit</th> </tr>
      </thead>
      <tbody>
        <xsl:apply-templates select="mm:unit"/>
      </tbody>
    </table>
  </xsl:template>

  <xsl:template match="mm:unit">
    <tr>
      <!-- mark the SI unit -->
      <xsl:if test="../@SIunit=@name">
        <xsl:attribute name="class">success</xsl:attribute>
      </xsl:if>
      <!-- unit name -->
      <td><code><xsl:value-of select="@name"/></code></td>
      <!-- outout conversion -->
      <td><code><xsl:value-of select="."/></code></td>
    </tr>
  </xsl:template>

</xsl:stylesheet>
