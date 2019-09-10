function [x, x0] = r_tls(y, A, L, delta)

%%%%%%%%%%%%%%%%%%%%%%% Lingfei Song 2019.9.2 %%%%%%%%%%%%%%%%%%%%%%%%
% A Solver for Quadratically Constrained Total Least Squares Problems
%                ---------------------------------            
%               |   min ||[E n]||_F^2             |
%               |   s.t. y + n = (A + E) * x      |
%               |        ||L * x||^2 <= \delta^2  | 
%                ---------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lb = 0;    % lower bound
x0 = rls(y, A, L, delta);
ub = norm(A * x0 - y)^2 / (1 + norm(x0)^2);    % upper bound
fprintf('>>> Initialized. Lower bound: %f, Upper bound: %f. \n', lb, ub);

% figure; mesh(reshape(x0, [15,15])); title('x0');

MAX_ITER = 500;
alpha = zeros(MAX_ITER,1);
for i = 1:MAX_ITER
    alpha(i) = (lb + ub) / 2;
    [~, beta] = sub_prog(A' * A - alpha(i) * eye(size(A,2)),  A' * y,  y' * y - alpha(i),  L,  delta);    % min ||A * x - y||^2 - alpha * (1 + ||x||^2),  s.t. ||L * x||^2 <= \delta^2
    fprintf('>>> Iteration: %d, alpha = %f. \n', i, alpha(i));
    if beta <= 0 
        ub = alpha(i);
    else
        lb = alpha(i);
    end
    if ub - lb <= 1E-6
        break
    end
end

[x, ~] = sub_prog(A' * A - ub * eye(size(A,2)),  A' * y,  y' * y - ub,  L,  delta);

% figure; plot(1:i, alpha(1:i)); xlabel('Iter'); ylabel('alpha');
% figure; mesh(reshape(x,15,15)); title('opt\_x');
% figure; plot(1:i, tmp(1:i)); title('tmp');
% figure; mesh(X(1:i, :)); title('X');


function [x] = rls(y, A, L, delta)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A Solver for Quadratically Constrained Least Squares Problems
%               --------------------------------
%              |  min ||A * x - y||^2           |
%              |  s.t. ||L * x||^2 <= \delta^2  |
%               --------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MAX_ITER = 1000;
lmd = zeros(MAX_ITER, 1);
lmd(1) = 200;
eta = 100;

H_A = A' * A;    % Hermitian matrix
H_L = L' * L;

x_0 = (H_A + lmd(1) * H_L)^(-1) * A' * y;
% eta = abs(0.001 * lmd(1) / (norm(L * x_0)^2 - delta^2));
for i = 2 : MAX_ITER
    lmd(i) = lmd(i-1) + eta * (norm(L * x_0)^2 - delta^2);
    x_1 = (H_A + lmd(i) * H_L)^(-1) * A' * y;
    if abs(norm(L * x_1) - delta) < delta * 1E-5 
        break;
    end
    x_0 = x_1;
    eta = eta * 1.05;
end

x = x_1;
% figure; plot(1:i, lmd(1:i)); xlabel('Iter'); ylabel('lmd');


function [x, beta] = sub_prog(A, b, c, L, delta)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A Solver for a Sub-Problem of the Constrained Total Least Squares
%               -----------------------------------
%              |  min x' * A * x - 2 * b' * x + c  |
%              |  s.t. x' * B * x <= \delta^2      |
%               -----------------------------------
% ATTENTION: A is not positive (semi)definite, therefore the above 
% program is not convex !!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[r, n] = size(L);    % rank of L

persistent B;
persistent V;

if isempty(B)
    B = L' * L;
    [V, ~] = eig(B);
end

if r == n
    [C, D] = s_diag_p(A, L);
else
    F = V(:,1:n-r);
    [C, D] = s_diag(A, L, F);
end

% fprintf('%f %f \n', A(5,5), A(10,10));
%----------------------------------------------------------------------
% making the change of variables x = C * z.
% making the change of variables z_j = sign(f_j) * sqrt(v_j), v_j >= 0.
%----------------------------------------------------------------------
f = C' * b;

lmd = diag(D(1:r, 1:r));
if lmd(r) > 0 && sum(f(1:r).^2 ./ lmd.^2) < delta^2
    opt_lmd = 0;
else
    %--------------------------------------------------------------------------------------
    % SEC "A unified convergence analysis of second order methods for secular equations."
    %--------------------------------------------------------------------------------------
    opt_lmd = min(lmd(r), 0) - 1E-5;
    while true
        G = sum(f(1:r).^2 ./ (lmd - opt_lmd).^2);
        grad_G = sum(2 * (lmd - opt_lmd) .* f(1:r).^2 ./ (lmd - opt_lmd).^4);
        opt_lmd = opt_lmd + 2 * (G^(-0.5) - 1/delta) / (G^(-1.5) * grad_G);
        if abs(G - delta^2) < 1E-10
            break;
        end
    end
end

v = zeros(size(L,2),1);
v(1 : r) = f(1:r).^2 ./ (lmd - opt_lmd).^2;
v(r + 1 : end) = f(r + 1 : end).^2;
z = sign(f) .* v.^(0.5);
x = C * z;
beta = x' * A * x - 2 * b' * x + c;



function [C, D] = s_diag(A, L, F)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% An algorithm for simultaneous diagonalization of two matrices, A and B.
% B := L' * L is positive semi-definite.
%           ------------------------------------------------------
%          |  C' * B * C = diag(1,1, ..., 1, 0, 0, ..., 0)        |
%          |  C' * A * C = diag(s_1, s_2, ..., s_r, 1, 1, ..., 1) |
%           ------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = size(A, 1); r = n - size(F, 2);
persistent M;
if isempty(M)
    M = L' * (L * L')^(-1);
end
S = [M - F * (F' * A * F)^(-1) * F' * A * M, F];
A = S' * A * S;
E = A(1 : r, 1 : r);
G = A(r + 1 : n, r + 1 : n);
[Q1, D1] = eig(E);
[Q2, D2] = eig(G);
Q2 = Q2 * diag(1 ./ diag(D2).^(0.5));

C = S * blkdiag(Q1, Q2);
D = blkdiag(D1, eye(size(D2)));


function [C, D] = s_diag_p(A, L)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% An algorithm for simultaneous diagonalization of two matrices, A and B.
% B = L' * L is positive definite.
%             ----------------------------------------------------
%            |  C' * B * C = diag(1,1,1, ..., 1,1)                |
%            |  C' * A * C = diag(s_1, s_2, s_3, ..., s_n-1, s_n) |
%             ----------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
persistent inv_L;
persistent inv_LT;
if isempty(inv_L)
    inv_L = L^(-1);
    inv_LT = (L')^(-1);
end

[V, D] = eig(inv_LT * A * inv_L);
% fprintf('%f %f \n', D(1,1), A(10,10));
C = inv_L * V;

