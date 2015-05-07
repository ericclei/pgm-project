function [grad, cov] = gradient(hp_mu, hp_cov, F, index, RF, A, R, E, alpha)
    grad = zeros(1, length(F));
    cov = zeros(length(F), length(F));
    for j = 1:size(A,1)
        Aj = A(j,:);
        expDot = exp(-Aj*F');
        grad = grad + (RF(index,j) - 1/(1+expDot)) * Aj;
        cov = cov - (expDot / (1 + expDot)^2) * (Aj' * Aj);
    end
    hp_cov_inv = hp_cov ^ -1;
    grad = grad - (F - hp_mu) * hp_cov_inv;
    cov = cov - hp_cov_inv;
    
    if ~isempty(E)
        for j = 1:size(E,1)
            Ej = E(j, :);
            grad = grad + alpha * (R(index, j) - Ej * F') * Ej;
            cov = cov - alpha * (Ej' * Ej);
        end
    end
    cov = - cov ^ -1;
end