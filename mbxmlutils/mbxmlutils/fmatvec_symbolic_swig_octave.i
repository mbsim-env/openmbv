%module fmatvec_symbolic_swig_octave

#pragma SWIG nowarn=373,374,365,366,367,368,371,362,509,503,305,315

%{
#include <fmatvec/types.h>
#include <fmatvec/fmatvec.h>
#include <fmatvec/ast.h>
#include <sstream>

using namespace fmatvec;
%}

namespace std {
  template<class T> class shared_ptr {};
}
%template(shared_ptr_AST_Vertex) std::shared_ptr<const fmatvec::AST::Vertex>;

%include <std_string.i>
%include <fmatvec/types.h>
%include <fmatvec/matrix.h>
%include <fmatvec/ast.h>
%include <fmatvec/var_fixed_general_matrix.h>
%include <fmatvec/var_vector.h>
%include <fmatvec/var_general_matrix.h>

%template(Dummy1) fmatvec::Matrix<fmatvec::General, fmatvec::Var, fmatvec::Fixed<1>,fmatvec::IndependentVariable>;
%template(IndependentVector) fmatvec::Vector<fmatvec::Var, fmatvec::IndependentVariable>;
%template(Dummy2) fmatvec::Matrix<fmatvec::General, fmatvec::Var, fmatvec::Fixed<1>,fmatvec::SymbolicExpression>;
%template(SymbolicVector) fmatvec::Vector<fmatvec::Var, fmatvec::SymbolicExpression>;
%template(SymbolicMatrix) fmatvec::Matrix<fmatvec::General, fmatvec::Var, fmatvec::Var, fmatvec::SymbolicExpression>;

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
}

%extend fmatvec::Vector<fmatvec::Var, fmatvec::SymbolicExpression> {
  std::string __str__() {
    std::stringstream str;
    str<<*$self;
    return str.str();
  }
}

%extend fmatvec::Matrix<fmatvec::General, fmatvec::Var, fmatvec::Var, fmatvec::SymbolicExpression> {
  std::string __str__() {
    std::stringstream str;
    str<<*$self;
    return str.str();
  }
}

%inline %{
  typedef fmatvec::IndependentVariable IS;
  typedef fmatvec::SymbolicExpression SS;
  typedef fmatvec::Vector<fmatvec::Var, fmatvec::IndependentVariable> IV;
  typedef fmatvec::Vector<fmatvec::Var, fmatvec::SymbolicExpression> SV;
  typedef fmatvec::Matrix<fmatvec::General, fmatvec::Var, fmatvec::Var, fmatvec::SymbolicExpression> SM;

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

  /***** operator * *****/

  SS op_scalar_mul_IndependentVariable(const double &a, const IS &b) { return a*b; }
  SS op_scalar_mul_SymbolicExpression(const double &a, const SS &b) { return a*b; }
  SV op_scalar_mul_IndependentVector(const double &a, const IV &b) { return a*b; }
  SV op_scalar_mul_SymbolicVector(const double &a, const SV &b) { return a*b; }
  SM op_scalar_mul_SymbolicMatrix(const double &a, const SM &b) { return a*b; }
  SS op_IndependentVariable_mul_scalar(const IS &a, const double &b) { return a*b; }
  SS op_IndependentVariable_mul_IndependentVariable(const IS &a, const IS &b) { return a*b; }
  SS op_IndependentVariable_mul_SymbolicExpression(const IS &a, const SS &b) { return a*b; }
  SV op_IndependentVariable_mul_IndependentVector(const IS &a, const IV &b) { return a*b; }
  SV op_IndependentVariable_mul_SymbolicVector(const IS &a, const SV &b) { return a*b; }
  SM op_IndependentVariable_mul_SymbolicMatrix(const IS &a, const SM &b) { return a*b; }
  SS op_SymbolicExpression_mul_scalar(const SS &a, const double &b) { return a*b; }
  SS op_SymbolicExpression_mul_IndependentVariable(const SS &a, const IS &b) { return a*b; }
  SS op_SymbolicExpression_mul_SymbolicExpression(const SS &a, const SS &b) { return a*b; }
  SV op_SymbolicExpression_mul_IndependentVector(const SS &a, const IV &b) { return a*b; }
  SV op_SymbolicExpression_mul_SymbolicVector(const SS &a, const SV &b) { return a*b; }
  SM op_SymbolicExpression_mul_SymbolicMatrix(const SS &a, const SM &b) { return a*b; }
  SV op_IndependentVector_mul_scalar(const IV &a, const double &b) { return a*b; }
  SV op_IndependentVector_mul_IndependentVariable(const IV &a, const IS &b) { return a*b; }
  SV op_IndependentVector_mul_SymbolicExpression(const IV &a, const SS &b) { return a*b; }
  SV op_SymbolicVector_mul_scalar(const SV &a, const double &b) { return a*b; }
  SV op_SymbolicVector_mul_IndependentVariable(const SV &a, const IS &b) { return a*b; }
  SV op_SymbolicVector_mul_SymbolicExpression(const SV &a, const SS &b) { return a*b; }
  SM op_SymbolicMatrix_mul_scalar(const SM &a, const double &b) { return a*b; }
  SM op_SymbolicMatrix_mul_IndependentVariable(const SM &a, const IS &b) { return a*b; }
  SM op_SymbolicMatrix_mul_SymbolicExpression(const SM &a, const SS &b) { return a*b; }
  SV op_SymbolicMatrix_mul_IndependentVector(const SM &a, const IV &b) { return a*b; }
  SV op_SymbolicMatrix_mul_SymbolicVector(const SM &a, const SV &b) { return a*b; }
  SM op_SymbolicMatrix_mul_SymbolicMatrix(const SM &a, const SM &b) { return a*b; }

  SM op_IndependentVariable_mul_matrix(const IS &a, const octave_value &b) { return a*toMat(b); }
  SM op_SymbolicExpression_mul_matrix(const SS &a, const octave_value &b) { return a*toMat(b); }
  SM op_matrix_mul_IndependentVariable(const octave_value &a, const IS &b) { return toMat(a)*b; }
  SM op_matrix_mul_SymbolicExpression(const octave_value &a, const SS &b) { return toMat(a)*b; }
  SV op_matrix_mul_SymbolicVector(const octave_value &a, const SV &b) { return toMat(a)*b; }
  SV op_matrix_mul_IndependentVector(const octave_value &a, const IV &b) { return toMat(a)*b; }
  SM op_matrix_mul_SymbolicMatrix(const octave_value &a, const SM &b) { return toMat(a)*b; }
  SM op_SymbolicMatrix_mul_matrix(const SM &a, const octave_value &b) { return a*toMat(b); }

  /***** operator / *****/

  SS op_IndependentVariable_div_scalar(const IS &a, const double &b) { return a/b; }
  SS op_SymbolicExpression_div_scalar(const SS &a, const double &b) { return a/b; }
  SV op_IndependentVector_div_scalar(const IV &a, const double &b) { return a/b; }
  SV op_SymbolicVector_div_scalar(const SV &a, const double &b) { return a/b; }
  SM op_SymbolicMatrix_div_scalar(const SM &a, const double &b) { return a/b; }
  SS op_scalar_div_IndependentVariable(const double &a, const IS &b) { return a/b; }
  SS op_IndependentVariable_div_IndependentVariable(const IS &a, const IS &b) { return a/b; }
  SS op_SymbolicExpression_div_IndependentVariable(const SS &a, const IS &b) { return a/b; }
  SV op_IndependentVector_div_IndependentVariable(const IV &a, const IS &b) { return a/b; }
  SV op_SymbolicVector_div_IndependentVariable(const SV &a, const IS &b) { return a/b; }
  SM op_SymbolicMatrix_div_IndependentVariable(const SM &a, const IS &b) { return a/b; }
  SS op_scalar_div_SymbolicExpression(const double &a, const SS &b) { return a/b; }
  SS op_IndependentVariable_div_SymbolicExpression(const IS &a, const SS &b) { return a/b; }
  SS op_SymbolicExpression_div_SymbolicExpression(const SS &a, const SS &b) { return a/b; }
  SV op_IndependentVector_div_SymbolicExpression(const IV &a, const SS &b) { return a/b; }
  SV op_SymbolicVector_div_SymbolicExpression(const SV &a, const SS &b) { return a/b; }
  SM op_SymbolicMatrix_div_SymbolicExpression(const SM &a, const SS &b) { return a/b; }

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
  SV op_IndependentVector_add_IndependentVector(const IV &a, const IV &b) { return a+b; }
  SV op_SymbolicVector_add_IndependentVector(const SV &a, const IV &b) { return a+b; }
  SV op_IndependentVector_add_SymbolicVector(const IV &a, const SV &b) { return a+b; }
  SV op_SymbolicVector_add_SymbolicVector(const SV &a, const SV &b) { return a+b; }
  SM op_SymbolicMatrix_add_SymbolicMatrix(const SM &a, const SM &b) { return a+b; }

  SV op_IndependentVector_add_matrix(const IV &a, const octave_value &b) { return a+toVec(b); }
  SV op_SymbolicVector_add_matrix(const SV &a, const octave_value &b) { return a+toVec(b); }
  SV op_matrix_add_IndependentVector(const octave_value &a, const IV &b) { return toVec(a)+b; }
  SV op_matrix_add_SymbolicVector(const octave_value &a, const SV &b) { return toVec(a)+b; }
  SM op_SymbolicMatrix_add_matrix(const SM &a, const octave_value &b) { return a+toMat(b); }
  SM op_matrix_add_SymbolicMatrix(const octave_value &a, const SM &b) { return toMat(a)+b; }

  /***** operator - *****/
 
  SS op_IndependentVariable_sub_scalar(const IS &a, const double &b) { return a-b; }
  SS op_SymbolicExpression_sub_scalar(const SS &a, const double &b) { return a-b; }
  SS op_scalar_sub_IndependentVariable(const double &a, const IS &b) { return a-b; }
  SS op_IndependentVariable_sub_IndependentVariable(const IS &a, const IS &b) { return a-b; }
  SS op_SymbolicExpression_sub_IndependentVariable(const SS &a, const IS &b) { return a-b; }
  SS op_scalar_sub_SymbolicExpression(const double &a, const SS &b) { return a-b; }
  SS op_IndependentVariable_sub_SymbolicExpression(const IS &a, const SS &b) { return a-b; }
  SS op_SymbolicExpression_sub_SymbolicExpression(const SS &a, const SS &b) { return a-b; }
  SV op_IndependentVector_sub_IndependentVector(const IV &a, const IV &b) { return a-b; }
  SV op_SymbolicVector_sub_IndependentVector(const SV &a, const IV &b) { return a-b; }
  SV op_IndependentVector_sub_SymbolicVector(const IV &a, const SV &b) { return a-b; }
  SV op_SymbolicVector_sub_SymbolicVector(const SV &a, const SV &b) { return a-b; }
  SM op_SymbolicMatrix_sub_SymbolicMatrix(const SM &a, const SM &b) { return a-b; }

  SV op_IndependentVector_sub_matrix(const IV &a, const octave_value &b) { return a-toVec(b); }
  SV op_SymbolicVector_sub_matrix(const SV &a, const octave_value &b) { return a-toVec(b); }
  SV op_matrix_sub_IndependentVector(const octave_value &a, const IV &b) { return toVec(a)-b; }
  SV op_matrix_sub_SymbolicVector(const octave_value &a, const SV &b) { return toVec(a)-b; }
  SM op_SymbolicMatrix_sub_matrix(const SM &a, const octave_value &b) { return a-toMat(b); }
  SM op_matrix_sub_SymbolicMatrix(const octave_value &a, const SM &b) { return toMat(a)-b; }

%}
