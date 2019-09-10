function [phi] = kernel2phi(X_SIZE, kernel)

K_SIZE = size(kernel, 1);

phi = zeros(X_SIZE^2, X_SIZE^2);

idx = 1;
for j = (K_SIZE - 1) / 2 + 1 : X_SIZE + (K_SIZE - 1) / 2
    for i = (K_SIZE - 1) / 2 + 1 : X_SIZE + (K_SIZE - 1) / 2
        tmp = zeros(X_SIZE + K_SIZE -1, X_SIZE + K_SIZE - 1);
        tmp(i-(K_SIZE-1)/2:i+(K_SIZE-1)/2, j-(K_SIZE-1)/2:j+(K_SIZE-1)/2) = kernel;
        tmp = tmp((K_SIZE - 1) / 2 + 1 : X_SIZE + (K_SIZE - 1) / 2, (K_SIZE - 1) / 2 + 1 : X_SIZE + (K_SIZE - 1) / 2);
        phi(idx, :) = tmp(:);
        idx = idx + 1;
    end
end