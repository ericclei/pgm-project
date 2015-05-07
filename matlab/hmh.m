function [F_new, post_mean_new, post_cov_new] = hmh(post_mean, post_cov, hp_mu, hp_cov, F, index, RF, A, R, E, alpha)
    F_new = mvnrnd(post_mean, post_cov);
    [grad, post_cov_new] = gradient(hp_mu, hp_cov, F_new, index, RF, A, R, E, alpha);
    post_mean_new = F_new + rand() / 5 * grad * post_cov_new;
    
%     F
%     post_mean
%     F_new
%     post_mean_new
    
    a1 = loglikelihood(hp_mu, hp_cov, F_new, index, RF, A, R, E, alpha)
    a2 = loglikelihood(hp_mu, hp_cov, F, index, RF, A, R, E, alpha)
    a3 = log(mvnpdf(F, post_mean_new, post_cov_new))
    a4 = log(mvnpdf(F_new, post_mean, post_cov))
    acceptance = a1 - a2;
    
    acceptance = acceptance + log(mvnpdf(F, post_mean_new, post_cov_new))...
        - log(mvnpdf(F_new, post_mean, post_cov));
    if rand() < exp(acceptance)
        F_new = F;
        post_mean_new = post_mean;
        post_cov_new = post_cov;
%         0
%     else
%         1
    end
end