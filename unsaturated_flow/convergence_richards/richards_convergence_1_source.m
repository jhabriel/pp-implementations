% Defining variables
syms x
syms y
syms t

% Analytical solution
psi = - t * x * (1-x) * y * (1-y) -1;

% Water retention curves
theta = 1/(1-psi);
krw = psi^2;

% Determining terms
grad_psi = [diff(psi,x);diff(psi,y)];
q = - krw * grad_psi;
div_q = diff(q(1),x) + diff(q(2),y);
partial_theta_partial_t = diff(theta,t);
f = partial_theta_partial_t + div_q;