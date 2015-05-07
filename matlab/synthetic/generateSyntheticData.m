clear
close all
home
%%
nEntities = 3;
N = ones(nEntities,1)*1e3;
D = ones(nEntities,1)*1e2;
L = 10; % latent dim
SIGMA = ones(nEntities)*1;
U = cell(nEntities,1); % entity latent features
V = cell(nEntities,1); % observed features latent features
MU = 3;
%%
m = Inf;
M = -Inf;
for k=1:nEntities
  u = randn(N(k), L)*sqrt(MU)/5 + sqrt(MU);
  u = u/sqrt(L);
  U{k} = u;
  m = min(m, min(u(:)));
  M = max(M, max(u(:)));
  V{k} = randn(D(k), L);
end
% for k=1:nEntities
%   u = U{k};
%   U{k} = (u - m) / (M - m);
% end
%%
R = cell(nEntities);
F = cell(nEntities,1);
for k=1:nEntities
  for l=k+1:nEntities
    r = mvnrnd(U{k}*U{l}', SIGMA(k,l)*ones(1,N(l)));
%     if any(r(:)<=0) 
%       r(r<=0)
%       error('nonpositive rating'); 
%     end
    % discretize
    r = round(r);
    r(r<1) = 1;
    r(r>5) = 5;
    R{k,l} = r;
    R{l,k} = R{k,l}';
  end
  F{k} = mvnrnd(U{k}*V{k}', SIGMA(k,k)*ones(1,D(k)));
end
%%
obsDensity = .05;
trainR = R;
testR = R;
for k=1:nEntities
  for l=k+1:nEntities
    mask = R{k,l}*0;
    mask(randperm(numel(mask), fix(obsDensity*numel(mask)))) = 1;
    trainR{k,l} = sparse(R{k,l}.*mask);
    trainR{l,k} = trainR{k,l}';
    testR{k,l} = R{k,l}.*(1-mask);
    testR{l,k} = testR{k,l}';
  end
end
%%
save synthetic D F L N nEntities trainR R SIGMA U V testR
%%
% load synthetic
%%
for k=1:nEntities
  for l=k+1:nEntities
    [i,j,val] = find(trainR{k,l});
    dump = [i j val];
    dlmwrite(sprintf('R.%d.%d.train.dat',k,l),dump);
    dlmwrite(sprintf('R.%d.%d.test.dat',k,l),testR{k,l});
  end
  dlmwrite(sprintf('F.%d.dat',k),F{k});
end
