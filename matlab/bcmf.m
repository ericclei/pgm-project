% Create Dataset
U_real = rand(10,3);
V_real = rand(10,3);
A_real = rand(5,3) - 0.5;
B_real = rand(5,3) - 0.5;

F = binornd(1, 1./(1.+exp(U_real*A_real')));
G = binornd(1, 1./(1.+exp(V_real*B_real')));

R = normrnd(U_real * V_real', 1);

% Learn
d = 1;
alpha = 1;
U = zeros(size(U_real,1), d);
hpU_mu = zeros(1, d);
hpU_cov = eye(d, d);

V = zeros(size(V_real,1), d);
hpV_mu = zeros(1, d);
hpV_cov = eye(d, d);

for i = 1:size(U_real,1)
    U(i,:) = mvnrnd(hpU_mu, hpU_cov);
    V(i,:) = mvnrnd(hpV_mu, hpV_cov);
end

A = zeros(size(A_real,1), d);
hpA_mu = zeros(1, d);
hpA_cov = eye(d, d);

B = zeros(size(B_real,1), d);
hpB_mu = zeros(1, d);
hpB_cov = eye(d, d);

for i = 1:size(A_real,1)
    A(i,:) = mvnrnd(hpA_mu, hpA_cov);
    B(i,:) = mvnrnd(hpB_mu, hpB_cov);
end

for i = 1:size(A,1)
    [grad, postA_cov] = gradient(hpA_mu, hpA_cov, A(i,:), i, F', U, [], [], alpha);
    postA_mean = A(i,:) - rand() / 5 * grad * postA_cov;
end

for i = 1:size(B,1)
    [grad, postB_cov] = gradient(hpB_mu, hpB_cov, B(i,:), i, G', V, [], [], alpha);
    postB_mean = B(i,:) - rand() / 5 * grad * postB_cov;
end

for i = 1:size(U,1)
    [grad, postU_cov] = gradient(hpU_mu, hpU_cov, U(i,:), i, F, A, R, V, alpha);
    postU_mean = U(i,:) - rand() / 5 * grad * postU_cov;
end

for i = 1:size(V,1)
    [grad, postV_cov] = gradient(hpV_mu, hpV_cov, V(i,:), i, G, B, R', U, alpha);
    postV_mean = V(i,:) - rand() / 5 * grad * postV_cov;
end

% for iter = 1:1
R_pred = zeros(size(R));
for iter = 1:50
    
    for i = 1:size(A,1)
        [A(i,:), postA_mean, postA_cov] = hmh(postA_mean, postA_cov, hpA_mu, hpA_cov, A(i,:), i, F', U, [], [], alpha);
    end
    for i = 1:size(B,1)
        [B(i,:), postB_mean, postB_cov] = hmh(postB_mean, postB_cov, hpB_mu, hpB_cov, B(i,:), i, G', V, [], [], alpha);
    end
    for i = 1:size(U,1)
        [U(i,:), postU_mean, postU_cov] = hmh(postU_mean, postU_cov, hpU_mu, hpU_cov, U(i,:), i, F, A, R, V, alpha);
    end
    for i = 1:size(V,1)
        [V(i,:), postV_mean, postV_cov] = hmh(postV_mean, postV_cov, hpV_mu, hpV_cov, V(i,:), i, G, B, R', U, alpha);
    end
    
    [hpU_mu, hpU_cov] = sampleHyperparameters(U);
    [hpA_mu, hpA_cov] = sampleHyperparameters(A);
    [hpB_mu, hpB_cov] = sampleHyperparameters(B);
    [hpV_mu, hpV_cov] = sampleHyperparameters(V);
%     U_err = sqrt(sum(sum((U_real - U).^2))/30)
%     V_err = sqrt(sum(sum((V_real - V).^2))/30)
    if iter > 20
        R_pred = R_pred + U*V';
        R_err = sqrt(sum(sum((R - R_pred / (iter - 20)).^2))/100)
    end
end
