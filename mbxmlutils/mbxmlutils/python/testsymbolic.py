import mbxmlutils._internal
import sympy

c=3.1
i=sympy.Dummy()
s=2*i

y=sympy.simplify(c+i+i*s)
print(mbxmlutils._internal._serializeFunction(y))

y=sympy.atan2(c, s)
print(mbxmlutils._internal._serializeFunction(y))

y=sympy.Heaviside(s)
print(mbxmlutils._internal._serializeFunction(y))

y=sympy.Min(c, s, sympy.simplify(i*i))
print(mbxmlutils._internal._serializeFunction(y))

y=sympy.Max(c, s, sympy.simplify(i*i))
print(mbxmlutils._internal._serializeFunction(y))

y=sympy.Piecewise((s, sympy.simplify(i>1)), (c*c, sympy.simplify(s<2)), (s*i, sympy.simplify(2<i)))
print(mbxmlutils._internal._serializeFunction(y))
