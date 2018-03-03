// module name
%module casadi_oct

// include casadi headers
%{
  #include <casadi/casadi.hpp>
%}

// handle std::string
%include "std_string.i"

// handle exceptions
%include "exception.i"

%exception {
  try {
    $action
  }
  catch(const std::exception &e) {
    std::stringstream str;
    str<<e.what()<<std::endl;
    SWIG_exception(SWIG_RuntimeError, str.str().c_str());
  }
  catch(...) {
    SWIG_exception(SWIG_RuntimeError, "Unknown exception");
  }
}

// disable some not working elements
%ignore casadi::Matrix<casadi::SXElem>::operator double;
%ignore casadi::Matrix<casadi::SXElem>::operator int;
%ignore casadi::Matrix<double        >::operator double;
%ignore casadi::Matrix<double        >::operator int;

%ignore casadi::Matrix<casadi::SXElem>::operator+;
%ignore casadi::Matrix<casadi::SXElem>::operator-;
%ignore casadi::Matrix<double        >::operator+;
%ignore casadi::Matrix<double        >::operator-;

// index operator
%extend casadi::Matrix<casadi::SXElem> {
  casadi::Matrix<casadi::SXElem> __paren__(int r, int c=1) const { return (*$self)(r-1, c-1); }

  void __paren_asgn__(int r,        const casadi::Matrix<casadi::SXElem> &x) { (*$self)(r-1)     =x(0); }
  void __paren_asgn__(int r,        double                                x) { (*$self)(r-1)     =x; }
  void __paren_asgn__(int r, int c, const casadi::Matrix<casadi::SXElem> &x) { (*$self)(r-1, c-1)=x(0, 0); }
  void __paren_asgn__(int r, int c, double                                x) { (*$self)(r-1, c-1)=x; }
};

%extend casadi::Matrix<double> {
  double __paren__(int r, int c=1) const { return (*$self)(r-1, c-1).scalar(); }

  void __paren_asgn__(int r,        double x) { (*$self)(r-1)     =x; }
  void __paren_asgn__(int r, int c, double x) { (*$self)(r-1, c-1)=x; }
};

// operators
%inline %{
  casadi::SX op_SX_uplus        (const casadi::SX &a                     ) { return a;    }
  casadi::SX op_SX_uminus       (const casadi::SX &a                     ) { return -a;   }
  casadi::SX op_SX_not          (const casadi::SX &a                     ) { return !a;   }
  casadi::SX op_SX_add_SX       (const casadi::SX &a, const casadi::SX &b) { return a+b;  }
  casadi::SX op_scalar_add_SX   (double            a, const casadi::SX &b) { return a+b;  }
  casadi::SX op_SX_add_scalar   (const casadi::SX &a, double            b) { return a+b;  }
  casadi::SX op_SX_sub_SX       (const casadi::SX &a, const casadi::SX &b) { return a-b;  }
  casadi::SX op_scalar_sub_SX   (double            a, const casadi::SX &b) { return a-b;  }
  casadi::SX op_SX_sub_scalar   (const casadi::SX &a, double            b) { return a-b;  }
  casadi::SX op_SX_mul_SX       (const casadi::SX &a, const casadi::SX &b) { return a*b;  }
  casadi::SX op_scalar_mul_SX   (double            a, const casadi::SX &b) { return a*b;  }
  casadi::SX op_SX_mul_scalar   (const casadi::SX &a, double            b) { return a*b;  }
  casadi::SX op_SX_div_SX       (const casadi::SX &a, const casadi::SX &b) { return a/b;  }
  casadi::SX op_scalar_div_SX   (double            a, const casadi::SX &b) { return a/b;  }
  casadi::SX op_SX_div_scalar   (const casadi::SX &a, double            b) { return a/b;  }
  casadi::SX op_SX_pow_SX       (const casadi::SX &a, const casadi::SX &b) { return pow(a, b); }
  casadi::SX op_scalar_pow_SX   (double            a, const casadi::SX &b) { return pow(a, b); }
  casadi::SX op_SX_pow_scalar   (const casadi::SX &a, double            b) { return pow(a, b); }
  casadi::SX op_SX_lt_SX        (const casadi::SX &a, const casadi::SX &b) { return a<b;  }
  casadi::SX op_scalar_lt_SX    (double            a, const casadi::SX &b) { return a<b;  }
  casadi::SX op_SX_lt_scalar    (const casadi::SX &a, double            b) { return a<b;  }
  casadi::SX op_SX_le_SX        (const casadi::SX &a, const casadi::SX &b) { return a<=b; }
  casadi::SX op_scalar_le_SX    (double            a, const casadi::SX &b) { return a<=b; }
  casadi::SX op_SX_le_scalar    (const casadi::SX &a, double            b) { return a<=b; }
  casadi::SX op_SX_gt_SX        (const casadi::SX &a, const casadi::SX &b) { return a>b;  }
  casadi::SX op_scalar_gt_SX    (double            a, const casadi::SX &b) { return a>b;  }
  casadi::SX op_SX_gt_scalar    (const casadi::SX &a, double            b) { return a>b;  }
  casadi::SX op_SX_ge_SX        (const casadi::SX &a, const casadi::SX &b) { return a>=b; }
  casadi::SX op_scalar_ge_SX    (double            a, const casadi::SX &b) { return a>=b; }
  casadi::SX op_SX_ge_scalar    (const casadi::SX &a, double            b) { return a>=b; }
  casadi::SX op_SX_eq_SX        (const casadi::SX &a, const casadi::SX &b) { return a==b; }
  casadi::SX op_scalar_eq_SX    (double            a, const casadi::SX &b) { return a==b; }
  casadi::SX op_SX_eq_scalar    (const casadi::SX &a, double            b) { return a==b; }
  casadi::SX op_SX_ne_SX        (const casadi::SX &a, const casadi::SX &b) { return a!=b; }
  casadi::SX op_scalar_ne_SX    (double            a, const casadi::SX &b) { return a!=b; }
  casadi::SX op_SX_ne_scalar    (const casadi::SX &a, double            b) { return a!=b; }
  casadi::SX op_SX_el_and_SX    (const casadi::SX &a, const casadi::SX &b) { return a&&b; }
  casadi::SX op_scalar_el_and_SX(double            a, const casadi::SX &b) { return a&&b; }
  casadi::SX op_SX_el_and_scalar(const casadi::SX &a, double            b) { return a&&b; }
  casadi::SX op_SX_el_or_SX     (const casadi::SX &a, const casadi::SX &b) { return a||b; }
  casadi::SX op_scalar_el_or_SX (double            a, const casadi::SX &b) { return a||b; }
  casadi::SX op_SX_el_or_scalar (const casadi::SX &a, double            b) { return a||b; }
  casadi::DM op_DM_uplus        (const casadi::DM &a                     ) { return a;    }
  casadi::DM op_DM_uminus       (const casadi::DM &a                     ) { return -a;   }
  casadi::DM op_DM_not          (const casadi::DM &a                     ) { return !a;   }
  casadi::DM op_DM_add_DM       (const casadi::DM &a, const casadi::DM &b) { return a+b;  }
  casadi::DM op_scalar_add_DM   (double            a, const casadi::DM &b) { return a+b;  }
  casadi::DM op_DM_add_scalar   (const casadi::DM &a, double            b) { return a+b;  }
  casadi::DM op_DM_sub_DM       (const casadi::DM &a, const casadi::DM &b) { return a-b;  }
  casadi::DM op_scalar_sub_DM   (double            a, const casadi::DM &b) { return a-b;  }
  casadi::DM op_DM_sub_scalar   (const casadi::DM &a, double            b) { return a-b;  }
  casadi::DM op_DM_mul_DM       (const casadi::DM &a, const casadi::DM &b) { return a*b;  }
  casadi::DM op_scalar_mul_DM   (double            a, const casadi::DM &b) { return a*b;  }
  casadi::DM op_DM_mul_scalar   (const casadi::DM &a, double            b) { return a*b;  }
  casadi::DM op_DM_div_DM       (const casadi::DM &a, const casadi::DM &b) { return a/b;  }
  casadi::DM op_scalar_div_DM   (double            a, const casadi::DM &b) { return a/b;  }
  casadi::DM op_DM_div_scalar   (const casadi::DM &a, double            b) { return a/b;  }
  casadi::DM op_DM_pow_DM       (const casadi::DM &a, const casadi::DM &b) { return pow(a, b); }
  casadi::DM op_scalar_pow_DM   (double            a, const casadi::DM &b) { return pow(a, b); }
  casadi::DM op_DM_pow_scalar   (const casadi::DM &a, double            b) { return pow(a, b); }
  casadi::DM op_DM_lt_DM        (const casadi::DM &a, const casadi::DM &b) { return a<b;  }
  casadi::DM op_scalar_lt_DM    (double            a, const casadi::DM &b) { return a<b;  }
  casadi::DM op_DM_lt_scalar    (const casadi::DM &a, double            b) { return a<b;  }
  casadi::DM op_DM_le_DM        (const casadi::DM &a, const casadi::DM &b) { return a<=b; }
  casadi::DM op_scalar_le_DM    (double            a, const casadi::DM &b) { return a<=b; }
  casadi::DM op_DM_le_scalar    (const casadi::DM &a, double            b) { return a<=b; }
  casadi::DM op_DM_gt_DM        (const casadi::DM &a, const casadi::DM &b) { return a>b;  }
  casadi::DM op_scalar_gt_DM    (double            a, const casadi::DM &b) { return a>b;  }
  casadi::DM op_DM_gt_scalar    (const casadi::DM &a, double            b) { return a>b;  }
  casadi::DM op_DM_ge_DM        (const casadi::DM &a, const casadi::DM &b) { return a>=b; }
  casadi::DM op_scalar_ge_DM    (double            a, const casadi::DM &b) { return a>=b; }
  casadi::DM op_DM_ge_scalar    (const casadi::DM &a, double            b) { return a>=b; }
  casadi::DM op_DM_eq_DM        (const casadi::DM &a, const casadi::DM &b) { return a==b; }
  casadi::DM op_scalar_eq_DM    (double            a, const casadi::DM &b) { return a==b; }
  casadi::DM op_DM_eq_scalar    (const casadi::DM &a, double            b) { return a==b; }
  casadi::DM op_DM_ne_DM        (const casadi::DM &a, const casadi::DM &b) { return a!=b; }
  casadi::DM op_scalar_ne_DM    (double            a, const casadi::DM &b) { return a!=b; }
  casadi::DM op_DM_ne_scalar    (const casadi::DM &a, double            b) { return a!=b; }
  casadi::DM op_DM_el_and_DM    (const casadi::DM &a, const casadi::DM &b) { return a&&b; }
  casadi::DM op_scalar_el_and_DM(double            a, const casadi::DM &b) { return a&&b; }
  casadi::DM op_DM_el_and_scalar(const casadi::DM &a, double            b) { return a&&b; }
  casadi::DM op_DM_el_or_DM     (const casadi::DM &a, const casadi::DM &b) { return a||b; }
  casadi::DM op_scalar_el_or_DM (double            a, const casadi::DM &b) { return a||b; }
  casadi::DM op_DM_el_or_scalar (const casadi::DM &a, double            b) { return a||b; }
  casadi::SX op_SX_add_DM       (const casadi::SX &a, const casadi::DM &b) { return a+casadi::SX(b);  }
  casadi::SX op_DM_add_SX       (const casadi::DM &a, const casadi::SX &b) { return casadi::SX(a)+b;  }
  casadi::SX op_SX_sub_DM       (const casadi::SX &a, const casadi::DM &b) { return a-casadi::SX(b);  }
  casadi::SX op_DM_sub_SX       (const casadi::DM &a, const casadi::SX &b) { return casadi::SX(a);    }
  casadi::SX op_SX_mul_DM       (const casadi::SX &a, const casadi::DM &b) { return a*casadi::SX(b);  }
  casadi::SX op_DM_mul_SX       (const casadi::DM &a, const casadi::SX &b) { return casadi::SX(a)*b;  }
  casadi::SX op_SX_div_DM       (const casadi::SX &a, const casadi::DM &b) { return a/casadi::SX(b);  }
  casadi::SX op_DM_div_SX       (const casadi::DM &a, const casadi::SX &b) { return casadi::SX(a)/b;  }
  casadi::SX op_SX_pow_DM       (const casadi::SX &a, const casadi::DM &b) { return pow(a, casadi::SX(b)); }
  casadi::SX op_DM_pow_SX       (const casadi::DM &a, const casadi::SX &b) { return pow(casadi::SX(a), b); }
  casadi::SX op_SX_lt_DM        (const casadi::SX &a, const casadi::DM &b) { return a<casadi::SX(b);  }
  casadi::SX op_DM_lt_SX        (const casadi::DM &a, const casadi::SX &b) { return casadi::SX(a)<b;  }
  casadi::SX op_SX_le_DM        (const casadi::SX &a, const casadi::DM &b) { return a<=casadi::SX(b); }
  casadi::SX op_DM_le_SX        (const casadi::DM &a, const casadi::SX &b) { return casadi::SX(a)<=b; }
  casadi::SX op_SX_gt_DM        (const casadi::SX &a, const casadi::DM &b) { return a>casadi::SX(b);  }
  casadi::SX op_DM_gt_SX        (const casadi::DM &a, const casadi::SX &b) { return casadi::SX(a)>b;  }
  casadi::SX op_SX_ge_DM        (const casadi::SX &a, const casadi::DM &b) { return a>=casadi::SX(b); }
  casadi::SX op_DM_ge_SX        (const casadi::DM &a, const casadi::SX &b) { return casadi::SX(a)>=b; }
  casadi::SX op_SX_eq_DM        (const casadi::SX &a, const casadi::DM &b) { return a==casadi::SX(b); }
  casadi::SX op_DM_eq_SX        (const casadi::DM &a, const casadi::SX &b) { return casadi::SX(a)==b; }
  casadi::SX op_SX_ne_DM        (const casadi::SX &a, const casadi::DM &b) { return a!=casadi::SX(b); }
  casadi::SX op_DM_ne_SX        (const casadi::DM &a, const casadi::SX &b) { return casadi::SX(a)=b;  }
  casadi::SX op_SX_el_and_DM    (const casadi::SX &a, const casadi::DM &b) { return a&&casadi::SX(b); }
  casadi::SX op_DM_el_and_SX    (const casadi::DM &a, const casadi::SX &b) { return casadi::SX(a)&&b; }
  casadi::SX op_SX_el_or_DM     (const casadi::SX &a, const casadi::DM &b) { return a||casadi::SX(b); }
  casadi::SX op_DM_el_or_SX     (const casadi::DM &a, const casadi::SX &b) { return casadi::SX(a)||b; }
%}

// concatication
%inline %{
  casadi::SX vertconcat_wrapper(const casadi::SX &a, const casadi::SX &b) { return casadi::SX::vertcat({a, b}); }
  casadi::SX horzconcat_wrapper(const casadi::SX &a, const casadi::SX &b) { return casadi::SX::horzcat({a, b}); }
%}

// print a human readable description in swig
%rename(__str__) casadi::PrintableObject<casadi::Matrix<casadi::SXElem> >::getRepresentation;
%rename(__str__) casadi::PrintableObject<casadi::Matrix<double        > >::getRepresentation;

// forward declaration for swig
namespace casadi {
  typedef casadi::Matrix<casadi::SXElem> SX;
  class Slice;
}

// include headers to wrap (including template instantations)
%include <casadi/core/printable_object.hpp>
%template(PrintableObject_Matrix_SXElem) casadi::PrintableObject<casadi::Matrix<casadi::SXElem> >;
%template(PrintableObject_Matrix_double) casadi::PrintableObject<casadi::Matrix<double        > >;

%include <casadi/core/sparsity_interface.hpp>
%template(SparsityInterface_Matrix_SXElem) casadi::SparsityInterface<casadi::Matrix<casadi::SXElem> >;
%template(SparsityInterface_Matrix_double) casadi::SparsityInterface<casadi::Matrix<double        > >;

%include <casadi/core/casadi_types.hpp>
%include <casadi/core/generic_matrix.hpp>
%template(GenericMatrix_Matrix_SXElem) casadi::GenericMatrix<casadi::Matrix<casadi::SXElem> >;
%template(GenericMatrix_Matrix_double) casadi::GenericMatrix<casadi::Matrix<double        > >;

%include <casadi/core/generic_expression.hpp>
%template(GenericExpression_Matrix_SXElem) casadi::GenericExpression<casadi::Matrix<casadi::SXElem> >;
%template(GenericExpression_Matrix_double) casadi::GenericExpression<casadi::Matrix<double        > >;

%include <casadi/core/matrix.hpp>
%template(SX) casadi::Matrix<casadi::SXElem>;
%template(DM) casadi::Matrix<double        >;

%include <casadi/core/calculus.hpp> // defines OP_SIN, ... used in @swig_ref/sin.m, ...
