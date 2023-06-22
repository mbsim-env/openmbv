function angles=invCardan(T)

  if T(1,3)<-1-1e-12 || T(1,3)>1+1e-12
    error('Argument of invCardan is not a rotation matrix (due to numerical errors)');
  end
  beta=asin(max(min(T(1,3), 1.0), -1.0));
  nenner=cos(beta);
  if abs(nenner)>1e-10
    alpha=atan2(-T(2,3),T(3,3));
    gamma=atan2(-T(1,2),T(1,1));
  else
    alpha=0;
    gamma=atan2(T(2,1),T(2,2));
  end
  angles=[alpha, beta, gamma];
