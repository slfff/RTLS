close all;
load('.\data\kernel.mat');

kernel = imresize(kernel, 0.5);

x = kernel(:) / sum(kernel(:));

% diary('log.txt');
% diary on;

file = fopen('log.txt', 'w');
% e_rtls = zeros(length(300:100:2000),1);
% e_rls = zeros(length(300:100:2000),1);
% for i = 1:10
% idx = 1;
for N = 2000:1000:15000
A = rand(N, length(x));
A(A >= 0.5) = 1;
A(A < 0.5) = 0;

y = A * x + randn(size(A,1),1) * 0.01;
A = A + randn(size(A)) * 0.01;

laplace = [-1 -1 -1; -1 8 -1; -1 -1 -1];
phi = kernel2phi(size(kernel,1), laplace);

delta = norm(phi * x);
L = phi;

[opt_x, x0] = r_tls(y, A, L, delta);

% e_rtls(idx) = norm(x-opt_x)/norm(x) + e_rtls(idx);
% e_rls(idx) = norm(x-x0)/norm(x) + e_rls(idx);
% idx = idx + 1;
fprintf(file, '%f %f\n', norm(x-x0)/norm(x), norm(x-opt_x)/norm(x));
end;
% end

save;
fclose(file);
% diary off;