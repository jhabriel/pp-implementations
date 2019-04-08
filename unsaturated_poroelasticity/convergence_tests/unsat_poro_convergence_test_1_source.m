% Determining the source term expression corresponding to the
% unsaturated Biot equations test #1

syms x     % first cartesian component
syms y     % second cartesian component
syms t     % time 

%% The displacement solution
u = [t*x*(1-x)*y*(1-y), -t*x*(1-x)*y*(1-y)];
p = -t*x*(1-x)*y*(1-y)-1;

%% Consitutive relationships
Sw = 1/(1-p);
krw = p^2;

%% Useful data
gradient_u = [diff(u(1),x),diff(u(1),y)
              diff(u(2),x),diff(u(2),y)];
divergence_u = diff(u(1),x) + diff(u(2),y); 
sw_times_p = Sw * p;


gradient_p = [diff(p,x),diff(p,y)];
gradient_sw_times_p = [diff(sw_times_p,x),diff(sw_times_p,y)];

%% Mechanical terms

% The strain tensor          
epsilon = 0.5 * (gradient_u + transpose(gradient_u));

% The effective stress
sigma_eff = 2*epsilon + divergence_u*eye(2);

% Divergence of the effective stress
div_sigma_eff = [diff(sigma_eff(1,1),x) + diff(sigma_eff(2,1),y),...
                 diff(sigma_eff(1,2),x) + diff(sigma_eff(2,2),y)];

% The momentum equation
momentum_eq = div_sigma_eff - gradient_sw_times_p;          

%% Flow terms

% Darcy's law
qw = - krw * gradient_p;

% Divergence of Darcy's velocity
div_qw = diff(qw(1),x) + diff(qw(2),y);

% Time derivative divergence of u
time_deriv_div_u = diff(divergence_u,t);

% Mass conservation equation
mass_eq = Sw * time_deriv_div_u + div_qw;

%% The source terms
F = transpose(momentum_eq);
f = mass_eq;