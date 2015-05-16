// module name
%module casadi

// include casadi headers
%{
  #include <casadi/core/sx/sx_element.hpp>
  #include <casadi/core/matrix/matrix.hpp>
  #include <casadi/core/function/sx_function.hpp>
%}

// import std stirng and pair support
%include "std_string.i"
%include "std_vector.i"
%include "exception.i"


// handle exceptions
%exception {
  try {
    $action
  }
  catch(const std::exception &e) {
    std::stringstream str;
    str<<"Exception: "<<std::endl
       <<e.what()<<std::endl;
    SWIG_exception(SWIG_RuntimeError, str.str().c_str());
  }
  catch(...) {
    SWIG_exception(SWIG_RuntimeError, "Unknown exception");
  }
}

// disable some not working elements
%ignore casadi::Matrix<casadi::SXElement>::get;
%ignore casadi::Matrix<casadi::SXElement>::getNZ;

%ignore casadi::Matrix<double           >::get;
%ignore casadi::Matrix<double           >::getNZ;

%ignore casadi::Matrix<casadi::SXElement>::operator-; // readadded later
%ignore casadi::Matrix<casadi::SXElement>::operator+; // readadded later

// index operator
%extend casadi::Matrix<casadi::SXElement> {
  casadi::Matrix<casadi::SXElement> __paren__(int r, int c=1) const { return $self->elem(r-1, c-1); }

  void __paren_asgn__(int r, const casadi::Matrix<casadi::SXElement> &x) { $self->elem(r-1)=x.elem(0); }
  void __paren_asgn__(int r, double x) { $self->elem(r-1)=x; }
  void __paren_asgn__(int r, int c, const casadi::Matrix<casadi::SXElement> &x) { $self->elem(r-1, c-1)=x.elem(0, 0); }
  void __paren_asgn__(int r, int c, double x) { $self->elem(r-1, c-1)=x; }
};

%extend casadi::Matrix<double> {
  double __paren__(int r, int c=1) const { return $self->elem(r-1, c-1); }

  void __paren_asgn__(int r, double x) { $self->elem(r-1)=x; }
  void __paren_asgn__(int r, int c, double x) { $self->elem(r-1, c-1)=x; }
};

// operators
%inline %{
  casadi::SX      op_SX_uplus              (const casadi::SX&      a                          ) { return a;    }
  casadi::SX      op_SX_uminus             (const casadi::SX&      a                          ) { return -a;   }
  casadi::SX      op_SX_not                (const casadi::SX&      a                          ) { return !a;   }
  casadi::SX      op_SX_add_SX             (const casadi::SX&      a, const casadi::SX&      b) { return a+b;  }
  casadi::SX      op_scalar_add_SX         (double                 a, const casadi::SX&      b) { return a+b;  }
  casadi::SX      op_SX_add_scalar         (const casadi::SX&      a, double                 b) { return a+b;  }
  casadi::SX      op_SX_sub_SX             (const casadi::SX&      a, const casadi::SX&      b) { return a-b;  }
  casadi::SX      op_scalar_sub_SX         (double                 a, const casadi::SX&      b) { return a-b;  }
  casadi::SX      op_SX_sub_scalar         (const casadi::SX&      a, double                 b) { return a-b;  }
  casadi::SX      op_SX_mul_SX             (const casadi::SX&      a, const casadi::SX&      b) { return a*b;  }
  casadi::SX      op_scalar_mul_SX         (double                 a, const casadi::SX&      b) { return a*b;  }
  casadi::SX      op_SX_mul_scalar         (const casadi::SX&      a, double                 b) { return a*b;  }
  casadi::SX      op_SX_div_SX             (const casadi::SX&      a, const casadi::SX&      b) { return a/b;  }
  casadi::SX      op_scalar_div_SX         (double                 a, const casadi::SX&      b) { return a/b;  }
  casadi::SX      op_SX_div_scalar         (const casadi::SX&      a, double                 b) { return a/b;  }
  casadi::SX      op_SX_pow_SX             (const casadi::SX&      a, const casadi::SX&      b) { return pow(a, b); }
  casadi::SX      op_scalar_pow_SX         (double                 a, const casadi::SX&      b) { return pow(a, b); }
  casadi::SX      op_SX_pow_scalar         (const casadi::SX&      a, double                 b) { return pow(a, b); }
  casadi::SX      op_SX_lt_SX              (const casadi::SX&      a, const casadi::SX&      b) { return a<b;  }
  casadi::SX      op_scalar_lt_SX          (double                 a, const casadi::SX&      b) { return a<b;  }
  casadi::SX      op_SX_lt_scalar          (const casadi::SX&      a, double                 b) { return a<b;  }
  casadi::SX      op_SX_le_SX              (const casadi::SX&      a, const casadi::SX&      b) { return a<=b; }
  casadi::SX      op_scalar_le_SX          (double                 a, const casadi::SX&      b) { return a<=b; }
  casadi::SX      op_SX_le_scalar          (const casadi::SX&      a, double                 b) { return a<=b; }
  casadi::SX      op_SX_gt_SX              (const casadi::SX&      a, const casadi::SX&      b) { return a>b;  }
  casadi::SX      op_scalar_gt_SX          (double                 a, const casadi::SX&      b) { return a>b;  }
  casadi::SX      op_SX_gt_scalar          (const casadi::SX&      a, double                 b) { return a>b;  }
  casadi::SX      op_SX_ge_SX              (const casadi::SX&      a, const casadi::SX&      b) { return a>=b; }
  casadi::SX      op_scalar_ge_SX          (double                 a, const casadi::SX&      b) { return a>=b; }
  casadi::SX      op_SX_ge_scalar          (const casadi::SX&      a, double                 b) { return a>=b; }
  casadi::SX      op_SX_eq_SX              (const casadi::SX&      a, const casadi::SX&      b) { return a==b; }
  casadi::SX      op_scalar_eq_SX          (double                 a, const casadi::SX&      b) { return a==b; }
  casadi::SX      op_SX_eq_scalar          (const casadi::SX&      a, double                 b) { return a==b; }
  casadi::SX      op_SX_ne_SX              (const casadi::SX&      a, const casadi::SX&      b) { return a!=b; }
  casadi::SX      op_scalar_ne_SX          (double                 a, const casadi::SX&      b) { return a!=b; }
  casadi::SX      op_SX_ne_scalar          (const casadi::SX&      a, double                 b) { return a!=b; }
  casadi::SX      op_SX_el_and_SX          (const casadi::SX&      a, const casadi::SX&      b) { return a&&b; }
  casadi::SX      op_scalar_el_and_SX      (double                 a, const casadi::SX&      b) { return a&&b; }
  casadi::SX      op_SX_el_and_scalar      (const casadi::SX&      a, double                 b) { return a&&b; }
  casadi::SX      op_SX_el_or_SX           (const casadi::SX&      a, const casadi::SX&      b) { return a||b; }
  casadi::SX      op_scalar_el_or_SX       (double                 a, const casadi::SX&      b) { return a||b; }
  casadi::SX      op_SX_el_or_scalar       (const casadi::SX&      a, double                 b) { return a||b; }
  casadi::DMatrix op_DMatrix_uplus         (const casadi::DMatrix& a                          ) { return a;    }
  casadi::DMatrix op_DMatrix_uminus        (const casadi::DMatrix& a                          ) { return -a;   }
  casadi::DMatrix op_DMatrix_not           (const casadi::DMatrix& a                          ) { return !a;   }
  casadi::DMatrix op_DMatrix_add_DMatrix   (const casadi::DMatrix& a, const casadi::DMatrix& b) { return a+b;  }
  casadi::DMatrix op_scalar_add_DMatrix    (double                 a, const casadi::DMatrix& b) { return a+b;  }
  casadi::DMatrix op_DMatrix_add_scalar    (const casadi::DMatrix& a, double                 b) { return a+b;  }
  casadi::DMatrix op_DMatrix_sub_DMatrix   (const casadi::DMatrix& a, const casadi::DMatrix& b) { return a-b;  }
  casadi::DMatrix op_scalar_sub_DMatrix    (double                 a, const casadi::DMatrix& b) { return a-b;  }
  casadi::DMatrix op_DMatrix_sub_scalar    (const casadi::DMatrix& a, double                 b) { return a-b;  }
  casadi::DMatrix op_DMatrix_mul_DMatrix   (const casadi::DMatrix& a, const casadi::DMatrix& b) { return a*b;  }
  casadi::DMatrix op_scalar_mul_DMatrix    (double                 a, const casadi::DMatrix& b) { return a*b;  }
  casadi::DMatrix op_DMatrix_mul_scalar    (const casadi::DMatrix& a, double                 b) { return a*b;  }
  casadi::DMatrix op_DMatrix_div_DMatrix   (const casadi::DMatrix& a, const casadi::DMatrix& b) { return a/b;  }
  casadi::DMatrix op_scalar_div_DMatrix    (double                 a, const casadi::DMatrix& b) { return a/b;  }
  casadi::DMatrix op_DMatrix_div_scalar    (const casadi::DMatrix& a, double                 b) { return a/b;  }
  casadi::DMatrix op_DMatrix_pow_DMatrix   (const casadi::DMatrix& a, const casadi::DMatrix& b) { return pow(a, b); }
  casadi::DMatrix op_scalar_pow_DMatrix    (double                 a, const casadi::DMatrix& b) { return pow(a, b); }
  casadi::DMatrix op_DMatrix_pow_scalar    (const casadi::DMatrix& a, double                 b) { return pow(a, b); }
  casadi::DMatrix op_DMatrix_lt_DMatrix    (const casadi::DMatrix& a, const casadi::DMatrix& b) { return a<b;  }
  casadi::DMatrix op_scalar_lt_DMatrix     (double                 a, const casadi::DMatrix& b) { return a<b;  }
  casadi::DMatrix op_DMatrix_lt_scalar     (const casadi::DMatrix& a, double                 b) { return a<b;  }
  casadi::DMatrix op_DMatrix_le_DMatrix    (const casadi::DMatrix& a, const casadi::DMatrix& b) { return a<=b; }
  casadi::DMatrix op_scalar_le_DMatrix     (double                 a, const casadi::DMatrix& b) { return a<=b; }
  casadi::DMatrix op_DMatrix_le_scalar     (const casadi::DMatrix& a, double                 b) { return a<=b; }
  casadi::DMatrix op_DMatrix_gt_DMatrix    (const casadi::DMatrix& a, const casadi::DMatrix& b) { return a>b;  }
  casadi::DMatrix op_scalar_gt_DMatrix     (double                 a, const casadi::DMatrix& b) { return a>b;  }
  casadi::DMatrix op_DMatrix_gt_scalar     (const casadi::DMatrix& a, double                 b) { return a>b;  }
  casadi::DMatrix op_DMatrix_ge_DMatrix    (const casadi::DMatrix& a, const casadi::DMatrix& b) { return a>=b; }
  casadi::DMatrix op_scalar_ge_DMatrix     (double                 a, const casadi::DMatrix& b) { return a>=b; }
  casadi::DMatrix op_DMatrix_ge_scalar     (const casadi::DMatrix& a, double                 b) { return a>=b; }
  casadi::DMatrix op_DMatrix_eq_DMatrix    (const casadi::DMatrix& a, const casadi::DMatrix& b) { return a==b; }
  casadi::DMatrix op_scalar_eq_DMatrix     (double                 a, const casadi::DMatrix& b) { return a==b; }
  casadi::DMatrix op_DMatrix_eq_scalar     (const casadi::DMatrix& a, double                 b) { return a==b; }
  casadi::DMatrix op_DMatrix_ne_DMatrix    (const casadi::DMatrix& a, const casadi::DMatrix& b) { return a!=b; }
  casadi::DMatrix op_scalar_ne_DMatrix     (double                 a, const casadi::DMatrix& b) { return a!=b; }
  casadi::DMatrix op_DMatrix_ne_scalar     (const casadi::DMatrix& a, double                 b) { return a!=b; }
  casadi::DMatrix op_DMatrix_el_and_DMatrix(const casadi::DMatrix& a, const casadi::DMatrix& b) { return a&&b; }
  casadi::DMatrix op_scalar_el_and_DMatrix (double                 a, const casadi::DMatrix& b) { return a&&b; }
  casadi::DMatrix op_DMatrix_el_and_scalar (const casadi::DMatrix& a, double                 b) { return a&&b; }
  casadi::DMatrix op_DMatrix_el_or_DMatrix (const casadi::DMatrix& a, const casadi::DMatrix& b) { return a||b; }
  casadi::DMatrix op_scalar_el_or_DMatrix  (double                 a, const casadi::DMatrix& b) { return a||b; }
  casadi::DMatrix op_DMatrix_el_or_scalar  (const casadi::DMatrix& a, double                 b) { return a||b; }
  casadi::SX      op_SX_add_DMatrix        (const casadi::SX&      a, const casadi::DMatrix& b) { return a+casadi::SX(b);  }
  casadi::SX      op_DMatrix_add_SX        (const casadi::DMatrix& a, const casadi::SX&      b) { return casadi::SX(a)+b;  }
  casadi::SX      op_SX_sub_DMatrix        (const casadi::SX&      a, const casadi::DMatrix& b) { return a-casadi::SX(b);  }
  casadi::SX      op_DMatrix_sub_SX        (const casadi::DMatrix& a, const casadi::SX&      b) { return casadi::SX(a);    }
  casadi::SX      op_SX_mul_DMatrix        (const casadi::SX&      a, const casadi::DMatrix& b) { return a*casadi::SX(b);  }
  casadi::SX      op_DMatrix_mul_SX        (const casadi::DMatrix& a, const casadi::SX&      b) { return casadi::SX(a)*b;  }
  casadi::SX      op_SX_div_DMatrix        (const casadi::SX&      a, const casadi::DMatrix& b) { return a/casadi::SX(b);  }
  casadi::SX      op_DMatrix_div_SX        (const casadi::DMatrix& a, const casadi::SX&      b) { return casadi::SX(a)/b;  }
  casadi::SX      op_SX_pow_DMatrix        (const casadi::SX&      a, const casadi::DMatrix& b) { return pow(a, casadi::SX(b)); }
  casadi::SX      op_DMatrix_pow_SX        (const casadi::DMatrix& a, const casadi::SX&      b) { return pow(casadi::SX(a), b); }
  casadi::SX      op_SX_lt_DMatrix         (const casadi::SX&      a, const casadi::DMatrix& b) { return a<casadi::SX(b);  }
  casadi::SX      op_DMatrix_lt_SX         (const casadi::DMatrix& a, const casadi::SX&      b) { return casadi::SX(a)<b;  }
  casadi::SX      op_SX_le_DMatrix         (const casadi::SX&      a, const casadi::DMatrix& b) { return a<=casadi::SX(b); }
  casadi::SX      op_DMatrix_le_SX         (const casadi::DMatrix& a, const casadi::SX&      b) { return casadi::SX(a)<=b; }
  casadi::SX      op_SX_gt_DMatrix         (const casadi::SX&      a, const casadi::DMatrix& b) { return a>casadi::SX(b);  }
  casadi::SX      op_DMatrix_gt_SX         (const casadi::DMatrix& a, const casadi::SX&      b) { return casadi::SX(a)>b;  }
  casadi::SX      op_SX_ge_DMatrix         (const casadi::SX&      a, const casadi::DMatrix& b) { return a>=casadi::SX(b); }
  casadi::SX      op_DMatrix_ge_SX         (const casadi::DMatrix& a, const casadi::SX&      b) { return casadi::SX(a)>=b; }
  casadi::SX      op_SX_eq_DMatrix         (const casadi::SX&      a, const casadi::DMatrix& b) { return a==casadi::SX(b); }
  casadi::SX      op_DMatrix_eq_SX         (const casadi::DMatrix& a, const casadi::SX&      b) { return casadi::SX(a)==b; }
  casadi::SX      op_SX_ne_DMatrix         (const casadi::SX&      a, const casadi::DMatrix& b) { return a!=casadi::SX(b); }
  casadi::SX      op_DMatrix_ne_SX         (const casadi::DMatrix& a, const casadi::SX&      b) { return casadi::SX(a)=b;  }
  casadi::SX      op_SX_el_and_DMatrix     (const casadi::SX&      a, const casadi::DMatrix& b) { return a&&casadi::SX(b); }
  casadi::SX      op_DMatrix_el_and_SX     (const casadi::DMatrix& a, const casadi::SX&      b) { return casadi::SX(a)&&b; }
  casadi::SX      op_SX_el_or_DMatrix      (const casadi::SX&      a, const casadi::DMatrix& b) { return a||casadi::SX(b); }
  casadi::SX      op_DMatrix_el_or_SX      (const casadi::DMatrix& a, const casadi::SX&      b) { return casadi::SX(a)||b; }
%}

// print a human readable description in swig
%rename(__str__) casadi::PrintableObject<casadi::Matrix<casadi::SXElement> >::getDescription;
%rename(__str__) casadi::PrintableObject<casadi::Matrix<double           > >::getDescription;

// forward declaration for swig
#define SWIG_OUTPUT(arg) OUTPUT
#define SWIG_INPUT(arg) INPUT
namespace casadi {
  typedef casadi::Matrix<casadi::SXElement> SX;
  enum SparsityType;
  class Sparsity;
  class Slice;
  class GenericType;
  class MX;
  class CodeGenerator;
  typedef GenericType::Dictionary Dictionary;
  typedef std::vector<Matrix<SXElement> > SXVector;
  typedef std::vector<std::vector<Matrix<SXElement> > > SXVectorVector;
  typedef std::vector<MX> MXVector;
  typedef std::vector<std::vector<MX> > MXVectorVector;
  enum opt_type;
  template<typename T> class IOSchemeVector;
}

// include headers to wrap (including template instantations)
%include <casadi/core/printable_object.hpp>
%template(PrintableObject_Matrix_SXElement) casadi::PrintableObject<casadi::Matrix<casadi::SXElement> >;
%template(PrintableObject_Matrix_double) casadi::PrintableObject<casadi::Matrix<double> >;
%template(PrintableObject_SharedObject) casadi::PrintableObject<casadi::SharedObject>;

%include <casadi/core/matrix/sparsity_interface.hpp>
%template(SparsityInterface_Matrix_SXElement) casadi::SparsityInterface<casadi::Matrix<casadi::SXElement> >;
%template(SparsityInterface_Matrix_double) casadi::SparsityInterface<casadi::Matrix<double> >;

%include <casadi/core/matrix/generic_matrix.hpp>
%template(GenericMatrix_Matrix_SXElement) casadi::GenericMatrix<casadi::Matrix<casadi::SXElement> >;
%template(GenericMatrix_Matrix_double) casadi::GenericMatrix<casadi::Matrix<double> >;

%include <casadi/core/matrix/generic_expression.hpp>
%template(GenericExpression_Matrix_SXElement) casadi::GenericExpression<casadi::Matrix<casadi::SXElement> >;
%template(GenericExpression_Matrix_double) casadi::GenericExpression<casadi::Matrix<double> >;

%include <casadi/core/matrix/matrix.hpp>
%template(SX) casadi::Matrix<casadi::SXElement>;
%template(DMatrix) casadi::Matrix<double>;
%extend casadi::Matrix<casadi::SXElement> {
  %template(SX) Matrix<double>;
};

%include <casadi/core/function/io_interface.hpp>
%template(IOInterface_Function) casadi::IOInterface<casadi::Function>;

%include <casadi/core/shared_object.hpp>

%include <casadi/core/options_functionality.hpp>

%include <casadi/core/function/function.hpp>

%include <casadi/core/function/sx_function.hpp>
