%module fmatvec_symbolic_swig_octave

#pragma SWIG nowarn=373,374,375,365,366,367,368,371,362,509,503,305,315

%{
#include <fmatvec/fmatvec.h>
#include <fmatvec/ast.h>
#include <sstream>
%}

namespace std {
  template<class T> class shared_ptr {};
}
%template(dummy_shared_ptr_AST_Vertex) std::shared_ptr<const fmatvec::AST::Vertex>;

// typemaps
typedef fmatvec::IndependentVariable IS;
typedef fmatvec::SymbolicExpression SS;
#define CREATESS SWIG_Octave_NewPointerObj(new fmatvec::SymbolicExpression($1), $descriptor(fmatvec::SymbolicExpression*), SWIG_POINTER_OWN |  0 )
%typemap(out) typename fmatvec::OperatorResult<IS    , IS    >::Type { $result = CREATESS; }
%typemap(out) typename fmatvec::OperatorResult<IS    , int   >::Type { $result = CREATESS; }
%typemap(out) typename fmatvec::OperatorResult<SS    , int   >::Type { $result = CREATESS; }
%typemap(out) typename fmatvec::OperatorResult<IS    , double>::Type { $result = CREATESS; }
%typemap(out) typename fmatvec::OperatorResult<SS    , double>::Type { $result = CREATESS; }
%typemap(out) typename fmatvec::OperatorResult<int   , IS    >::Type { $result = CREATESS; }
%typemap(out) typename fmatvec::OperatorResult<double, IS    >::Type { $result = CREATESS; }
%typemap(out) typename fmatvec::OperatorResult<IS    , IS    >::Type { $result = CREATESS; }
%typemap(out) typename fmatvec::OperatorResult<SS    , IS    >::Type { $result = CREATESS; }
%typemap(out) typename fmatvec::OperatorResult<int   , SS    >::Type { $result = CREATESS; }
%typemap(out) typename fmatvec::OperatorResult<double, SS    >::Type { $result = CREATESS; }
%typemap(out) typename fmatvec::OperatorResult<IS    , SS    >::Type { $result = CREATESS; }
%typemap(out) typename fmatvec::OperatorResult<SS    , SS    >::Type { $result = CREATESS; }

// transpose
%rename(__hermitian__) fmatvec::Matrix<fmatvec::General, fmatvec::Var, fmatvec::Var, fmatvec::SymbolicExpression>::T;

%ignore fmatvec::Vector<fmatvec::Var, fmatvec::IndependentVariable>::operator();
%ignore fmatvec::Vector<fmatvec::Var, fmatvec::SymbolicExpression>::operator();
%ignore fmatvec::Matrix<fmatvec::General, fmatvec::Var, fmatvec::Var, fmatvec::SymbolicExpression>::operator();
%feature("valuewrapper") fmatvec::Vector<fmatvec::Var, fmatvec::IndependentVariable>;
%feature("valuewrapper") fmatvec::Vector<fmatvec::Var, fmatvec::SymbolicExpression>;
%feature("valuewrapper") fmatvec::Matrix<fmatvec::General, fmatvec::Var, fmatvec::Var, fmatvec::SymbolicExpression>;

%import <std_string.i>
%include <fmatvec/types.h>
%import <fmatvec/matrix.h>
%include <fmatvec/ast.h>
%import <fmatvec/var_fixed_general_matrix.h>
%import <fmatvec/var_vector.h>
%import <fmatvec/var_general_matrix.h>
%import <fmatvec/linear_algebra.h>

%template(dummy_matrix_general_var_fixed_indep) fmatvec::Matrix<fmatvec::General, fmatvec::Var, fmatvec::Fixed<1>,fmatvec::IndependentVariable>;
%template(VectorIndep) fmatvec::Vector<fmatvec::Var, fmatvec::IndependentVariable>;
%template(dummy_matrix_general_fixed_sym) fmatvec::Matrix<fmatvec::General, fmatvec::Var, fmatvec::Fixed<1>,fmatvec::SymbolicExpression>;
%template(VectorSym) fmatvec::Vector<fmatvec::Var, fmatvec::SymbolicExpression>;
%template(MatrixSym) fmatvec::Matrix<fmatvec::General, fmatvec::Var, fmatvec::Var, fmatvec::SymbolicExpression>;

%extend fmatvec::IndependentVariable {
  std::string __str__() {
    std::stringstream str;
    str<<*$self;
    return str.str();
  }
}

%extend fmatvec::SymbolicExpression {
  std::string __str__() {
    std::stringstream str;
    str<<*$self;
    return str.str();
  }
}

%extend fmatvec::Vector<fmatvec::Var, fmatvec::IndependentVariable> {
  std::string __str__() {
    std::stringstream str;
    str<<*$self;
    return str.str();
  }
  IndependentVariable& __paren__(int i) { return (*$self)(i-1); }
  void __paren_asgn__(int i, const fmatvec::IndependentVariable &x) { (*$self)(i-1)=x; }
}

%extend fmatvec::Vector<fmatvec::Var, fmatvec::SymbolicExpression> {
  std::string __str__() {
    std::stringstream str;
    str<<*$self;
    return str.str();
  }
  SymbolicExpression& __paren__(int i) { return (*$self)(i-1); }
  void __paren_asgn__(int i, const fmatvec::SymbolicExpression &x) { (*$self)(i-1)=x; }
  void __paren_asgn__(int i, const double &x) { (*$self)(i-1)=x; }
}

%extend fmatvec::Matrix<fmatvec::General, fmatvec::Var, fmatvec::Var, fmatvec::SymbolicExpression> {
  std::string __str__() {
    std::stringstream str;
    str<<*$self;
    return str.str();
  }
  SymbolicExpression& __paren__(int r, int c) { return (*$self)(r-1,c-1); }
  void __paren_asgn__(int r, int c, const fmatvec::SymbolicExpression &x) { (*$self)(r-1,c-1)=x; }
  void __paren_asgn__(int r, int c, const double &x) { (*$self)(r-1,c-1)=x; }
}

%{
  // helper functions

  fmatvec::MatV toMat(const octave_value &x) {
    ::Matrix m=x.matrix_value();
    int rows=m.rows();
    int cols=m.cols();
    fmatvec::MatV ret(rows, cols);
    for(int r=0; r<rows; r++)
      for(int c=0; c<cols; c++)
        ret(r,c)=m(r,c);
    return ret;
  }

  fmatvec::VecV toVec(const octave_value &x) {
    ::Matrix m=x.matrix_value();
    int rows=m.rows();
    if(m.cols()!=1)
      std::runtime_error("Matrix contains more than 1 column, cannot convert to vector.");
    fmatvec::VecV ret(rows);
    for(int r=0; r<rows; r++)
      ret(r)=m(r);
    return ret;
  }

  octave_value_list callBuiltin(const char* name, const octave_value_list &arg, int n=1) {
#if SWIG_OCTAVE_PREREQ(4,0,0)
    auto func=octave::interpreter::the_interpreter()->get_symbol_table().builtin_find(name).function_value();
#else
    auto func=symbol_table::builtin_find(name).function_value();
#endif
    return feval(func, arg, n);
  }
%}

%inline %{
  typedef fmatvec::IndependentVariable IS;
  typedef fmatvec::SymbolicExpression SS;
  typedef fmatvec::Vector<fmatvec::Var, fmatvec::IndependentVariable> IV;
  typedef fmatvec::Vector<fmatvec::Var, fmatvec::SymbolicExpression> SV;
  typedef fmatvec::Matrix<fmatvec::General, fmatvec::Var, fmatvec::Var, fmatvec::SymbolicExpression> SM;

  /***** operator * *****/

  SS op_scalar_mul_IndependentVariable(const double &a, const IS &b) { return a*b; }
  SS op_scalar_mul_SymbolicExpression(const double &a, const SS &b) { return a*b; }
  SV op_scalar_mul_VectorIndep(const double &a, const IV &b) { return a*b; }
  SV op_scalar_mul_VectorSym(const double &a, const SV &b) { return a*b; }
  SM op_scalar_mul_MatrixSym(const double &a, const SM &b) { return a*b; }
  SS op_IndependentVariable_mul_scalar(const IS &a, const double &b) { return a*b; }
  SS op_IndependentVariable_mul_IndependentVariable(const IS &a, const IS &b) { return a*b; }
  SS op_IndependentVariable_mul_SymbolicExpression(const IS &a, const SS &b) { return a*b; }
  SV op_IndependentVariable_mul_VectorIndep(const IS &a, const IV &b) { return a*b; }
  SV op_IndependentVariable_mul_VectorSym(const IS &a, const SV &b) { return a*b; }
  SM op_IndependentVariable_mul_MatrixSym(const IS &a, const SM &b) { return a*b; }
  SS op_SymbolicExpression_mul_scalar(const SS &a, const double &b) { return a*b; }
  SS op_SymbolicExpression_mul_IndependentVariable(const SS &a, const IS &b) { return a*b; }
  SS op_SymbolicExpression_mul_SymbolicExpression(const SS &a, const SS &b) { return a*b; }
  SV op_SymbolicExpression_mul_VectorIndep(const SS &a, const IV &b) { return a*b; }
  SV op_SymbolicExpression_mul_VectorSym(const SS &a, const SV &b) { return a*b; }
  SM op_SymbolicExpression_mul_MatrixSym(const SS &a, const SM &b) { return a*b; }
  SV op_VectorIndep_mul_scalar(const IV &a, const double &b) { return a*b; }
  SV op_VectorIndep_mul_IndependentVariable(const IV &a, const IS &b) { return a*b; }
  SV op_VectorIndep_mul_SymbolicExpression(const IV &a, const SS &b) { return a*b; }
  SV op_VectorSym_mul_scalar(const SV &a, const double &b) { return a*b; }
  SV op_VectorSym_mul_IndependentVariable(const SV &a, const IS &b) { return a*b; }
  SV op_VectorSym_mul_SymbolicExpression(const SV &a, const SS &b) { return a*b; }
  SM op_MatrixSym_mul_scalar(const SM &a, const double &b) { return a*b; }
  SM op_MatrixSym_mul_IndependentVariable(const SM &a, const IS &b) { return a*b; }
  SM op_MatrixSym_mul_SymbolicExpression(const SM &a, const SS &b) { return a*b; }
  SV op_MatrixSym_mul_VectorIndep(const SM &a, const IV &b) { return a*b; }
  SV op_MatrixSym_mul_VectorSym(const SM &a, const SV &b) { return a*b; }
  SM op_MatrixSym_mul_MatrixSym(const SM &a, const SM &b) { return a*b; }

  SM op_IndependentVariable_mul_matrix(const IS &a, const octave_value &b) { return a*toMat(b); }
  SM op_SymbolicExpression_mul_matrix(const SS &a, const octave_value &b) { return a*toMat(b); }
  SM op_matrix_mul_IndependentVariable(const octave_value &a, const IS &b) { return toMat(a)*b; }
  SM op_matrix_mul_SymbolicExpression(const octave_value &a, const SS &b) { return toMat(a)*b; }
  SV op_matrix_mul_VectorSym(const octave_value &a, const SV &b) { return toMat(a)*b; }
  SV op_matrix_mul_VectorIndep(const octave_value &a, const IV &b) { return toMat(a)*b; }
  SM op_matrix_mul_MatrixSym(const octave_value &a, const SM &b) { return toMat(a)*b; }
  SM op_MatrixSym_mul_matrix(const SM &a, const octave_value &b) { return a*toMat(b); }

  /***** operator / *****/

  SS op_IndependentVariable_div_scalar(const IS &a, const double &b) { return a/b; }
  SS op_SymbolicExpression_div_scalar(const SS &a, const double &b) { return a/b; }
  SV op_VectorIndep_div_scalar(const IV &a, const double &b) { return a/b; }
  SV op_VectorSym_div_scalar(const SV &a, const double &b) { return a/b; }
  SM op_MatrixSym_div_scalar(const SM &a, const double &b) { return a/b; }
  SS op_scalar_div_IndependentVariable(const double &a, const IS &b) { return a/b; }
  SS op_IndependentVariable_div_IndependentVariable(const IS &a, const IS &b) { return a/b; }
  SS op_SymbolicExpression_div_IndependentVariable(const SS &a, const IS &b) { return a/b; }
  SV op_VectorIndep_div_IndependentVariable(const IV &a, const IS &b) { return a/b; }
  SV op_VectorSym_div_IndependentVariable(const SV &a, const IS &b) { return a/b; }
  SM op_MatrixSym_div_IndependentVariable(const SM &a, const IS &b) { return a/b; }
  SS op_scalar_div_SymbolicExpression(const double &a, const SS &b) { return a/b; }
  SS op_IndependentVariable_div_SymbolicExpression(const IS &a, const SS &b) { return a/b; }
  SS op_SymbolicExpression_div_SymbolicExpression(const SS &a, const SS &b) { return a/b; }
  SV op_VectorIndep_div_SymbolicExpression(const IV &a, const SS &b) { return a/b; }
  SV op_VectorSym_div_SymbolicExpression(const SV &a, const SS &b) { return a/b; }
  SM op_MatrixSym_div_SymbolicExpression(const SM &a, const SS &b) { return a/b; }

  SM op_matrix_div_IndependentVariable(const octave_value &a, const IS &b) { return toMat(a)/b; }
  SM op_matrix_div_SymbolicExpression(const octave_value &a, const SS &b) { return toMat(a)/b; }

  /***** operator + *****/
 
  SS op_IndependentVariable_add_scalar(const IS &a, const double &b) { return a+b; }
  SS op_SymbolicExpression_add_scalar(const SS &a, const double &b) { return a+b; }
  SS op_scalar_add_IndependentVariable(const double &a, const IS &b) { return a+b; }
  SS op_IndependentVariable_add_IndependentVariable(const IS &a, const IS &b) { return a+b; }
  SS op_SymbolicExpression_add_IndependentVariable(const SS &a, const IS &b) { return a+b; }
  SS op_scalar_add_SymbolicExpression(const double &a, const SS &b) { return a+b; }
  SS op_IndependentVariable_add_SymbolicExpression(const IS &a, const SS &b) { return a+b; }
  SS op_SymbolicExpression_add_SymbolicExpression(const SS &a, const SS &b) { return a+b; }
  SV op_VectorIndep_add_VectorIndep(const IV &a, const IV &b) { return a+b; }
  SV op_VectorSym_add_VectorIndep(const SV &a, const IV &b) { return a+b; }
  SV op_VectorIndep_add_VectorSym(const IV &a, const SV &b) { return a+b; }
  SV op_VectorSym_add_VectorSym(const SV &a, const SV &b) { return a+b; }
  SM op_MatrixSym_add_MatrixSym(const SM &a, const SM &b) { return a+b; }

  SV op_VectorIndep_add_matrix(const IV &a, const octave_value &b) { return a+toVec(b); }
  SV op_VectorSym_add_matrix(const SV &a, const octave_value &b) { return a+toVec(b); }
  SV op_matrix_add_VectorIndep(const octave_value &a, const IV &b) { return toVec(a)+b; }
  SV op_matrix_add_VectorSym(const octave_value &a, const SV &b) { return toVec(a)+b; }
  SM op_MatrixSym_add_matrix(const SM &a, const octave_value &b) { return a+toMat(b); }
  SM op_matrix_add_MatrixSym(const octave_value &a, const SM &b) { return toMat(a)+b; }

  /***** operator - *****/
 
  SS op_IndependentVariable_sub_scalar(const IS &a, const double &b) { return a-b; }
  SS op_SymbolicExpression_sub_scalar(const SS &a, const double &b) { return a-b; }
  SS op_scalar_sub_IndependentVariable(const double &a, const IS &b) { return a-b; }
  SS op_IndependentVariable_sub_IndependentVariable(const IS &a, const IS &b) { return a-b; }
  SS op_SymbolicExpression_sub_IndependentVariable(const SS &a, const IS &b) { return a-b; }
  SS op_scalar_sub_SymbolicExpression(const double &a, const SS &b) { return a-b; }
  SS op_IndependentVariable_sub_SymbolicExpression(const IS &a, const SS &b) { return a-b; }
  SS op_SymbolicExpression_sub_SymbolicExpression(const SS &a, const SS &b) { return a-b; }
  SV op_VectorIndep_sub_VectorIndep(const IV &a, const IV &b) { return a-b; }
  SV op_VectorSym_sub_VectorIndep(const SV &a, const IV &b) { return a-b; }
  SV op_VectorIndep_sub_VectorSym(const IV &a, const SV &b) { return a-b; }
  SV op_VectorSym_sub_VectorSym(const SV &a, const SV &b) { return a-b; }
  SM op_MatrixSym_sub_MatrixSym(const SM &a, const SM &b) { return a-b; }

  SV op_VectorIndep_sub_matrix(const IV &a, const octave_value &b) { return a-toVec(b); }
  SV op_VectorSym_sub_matrix(const SV &a, const octave_value &b) { return a-toVec(b); }
  SV op_matrix_sub_VectorIndep(const octave_value &a, const IV &b) { return toVec(a)-b; }
  SV op_matrix_sub_VectorSym(const octave_value &a, const SV &b) { return toVec(a)-b; }
  SM op_MatrixSym_sub_matrix(const SM &a, const octave_value &b) { return a-toMat(b); }
  SM op_matrix_sub_MatrixSym(const octave_value &a, const SM &b) { return toMat(a)-b; }

  /***** operator - *****/
  SS op_IndependentVariable_uminus(const IS &a) { return -a; }
  SS op_SymbolicExpression_uminus(const SS &a) { return -a; }
  SV op_VectorIndep_uminus(const IV &a) { return -a; }
  SV op_VectorSym_uminus(const SV &a) { return -a; }
  SM op_MatrixSym_uminus(const SM &a) { return -a; }

  /***** operator ^ *****/
 
  SS op_scalar_pow_IndependentVariable(const double &a, const IS &b) { return pow(a,b); }
  SS op_scalar_pow_SymbolicExpression(const double &a, const SS &b) { return pow(a,b); }
  SS op_IndependentVariable_pow_scalar(const IS &a, const double &b) { return pow(a,b); }
  SS op_IndependentVariable_pow_IndependentVariable(const IS &a, const IS &b) { return pow(a,b); }
  SS op_IndependentVariable_pow_SymbolicExpression(const IS &a, const SS &b) { return pow(a,b); }
  SS op_SymbolicExpression_pow_scalar(const SS &a, const double &b) { return pow(a,b); }
  SS op_SymbolicExpression_pow_IndependentVariable(const SS &a, const IS &b) { return pow(a,b); }
  SS op_SymbolicExpression_pow_SymbolicExpression(const SS &a, const SS &b) { return pow(a,b); }

  namespace fmatvec {
    SymbolicExpression pow(const double &a, const SymbolicExpression &b) { return pow(SS(a),b); }
    SymbolicExpression pow(const SymbolicExpression &a, const double &b) { return pow(a,SS(b)); }
    octave_value       pow(const octave_value a, const octave_value b) { return callBuiltin("power", octave_value_list(a).append(b))(0); }
    SymbolicExpression pow(const SymbolicExpression &a, const int &b) { return pow(a,SS(b)); }
    SymbolicExpression pow(const double &a, const int &b) { return pow(a,SS(b)); }
    octave_value       log(const octave_value a) { return callBuiltin("log", a)(0); }
    octave_value       sqrt(const octave_value a) { return callBuiltin("sqrt", a)(0); }
    octave_value       sin(const octave_value a) { return callBuiltin("sin", a)(0); }
    octave_value       cos(const octave_value a) { return callBuiltin("cos", a)(0); }
    octave_value       tan(const octave_value a) { return callBuiltin("tan", a)(0); }
    octave_value       sinh(const octave_value a) { return callBuiltin("sinh", a)(0); }
    octave_value       cosh(const octave_value a) { return callBuiltin("cosh", a)(0); }
    octave_value       tanh(const octave_value a) { return callBuiltin("tanh", a)(0); }
    octave_value       asin(const octave_value a) { return callBuiltin("asin", a)(0); }
    octave_value       acos(const octave_value a) { return callBuiltin("acos", a)(0); }
    octave_value       atan(const octave_value a) { return callBuiltin("atan", a)(0); }
    octave_value       asinh(const octave_value a) { return callBuiltin("asinh", a)(0); }
    octave_value       acosh(const octave_value a) { return callBuiltin("acosh", a)(0); }
    octave_value       atanh(const octave_value a) { return callBuiltin("atanh", a)(0); }
    octave_value       exp(const octave_value a) { return callBuiltin("exp", a)(0); }
    octave_value       sign(const octave_value a) { return callBuiltin("sign", a)(0); }
    octave_value       abs(const octave_value a) { return callBuiltin("abs", a)(0); }

    SymbolicExpression norm(const IV &a) { return nrm2(a); }
    SymbolicExpression norm(const SV &a) { return nrm2(a); }
    octave_value       norm(const octave_value a) { return callBuiltin("norm", a)(0); }
  }

%}
