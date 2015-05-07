function [hp_mu, hp_cov] = sampleHyperparameters(F)
    FBar = mean(F);
    CovBar = cov(F);
    d = size(F, 2);
    n = size(F, 1);
    
    Psi = (n-1) * CovBar + eye(d,d) + FBar'*FBar*(n/(n + 1));
    Psi = inv(Psi);
    hp_cov = wishrnd(Psi, d + n);
    hp_mu = mvnrnd((n/(n+1)) * FBar, hp_cov * (1/(1+n)));