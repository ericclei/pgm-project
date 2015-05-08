function likelihood = loglikelihood(hp_mu, hp_cov, F, index, RF, A, R, E, alpha)
    likelihood = 0;
    for j = 1:size(A,1)
        Aj = A(j,:);
        dot = Aj*F';
        likelihood = likelihood + (RF(index, j) * dot - log(1 + exp(dot)));
    end
    likelihood = likelihood - 0.5 * (F - hp_mu) * (hp_cov \ (F - hp_mu)');
    
    if length(E) > 0
        for j = 1:size(E,1)
            Ej = E(j, :);
            likelihood = likelihood - 0.5 * alpha * (R(index, j) - Ej*F')^2;
        end
    end
end