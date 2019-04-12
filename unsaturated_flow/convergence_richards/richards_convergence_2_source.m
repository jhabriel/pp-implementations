% Defining variables
syms x
syms y
syms t

% Analytical solution
psi = - t * x * (1-x) * y * (1-y) -1;

% van Genuchten parameters
alpha = 0.04;
n = 2;
m = 1-(1/n);
theta_s = 0.4;
theta_r = 0.1;

% Water retention curves
theta = (theta_s-theta_r)/(1 + (alpha*abs(psi))^n)^m + theta_s;

krw = (1 - (alpha*abs(psi))^(n-1) * (1 + (alpha*abs(psi))^n)^(-m))^2/...
      (1 + (alpha*abs(psi))^n)^(m/2);

C = (-m*n*psi*(theta_s-theta_r)*(alpha*abs(psi))^n)/...
    (abs(psi)^2 * (1 + (alpha*abs(psi))^n)^(m+1));  

% Determining terms
grad_psi = [diff(psi,x);diff(psi,y)];
q = - krw * grad_psi;
div_q = diff(q(1),x) + diff(q(2),y);
partial_theta_partial_t = diff(theta,t);
f = partial_theta_partial_t + div_q;