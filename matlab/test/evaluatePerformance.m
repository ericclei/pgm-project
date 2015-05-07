%% testing feature-enriched collective mf

clear
close all
home
load RTest

%%
U0 = importU('U.0.(0).dat', 0);
U1 = importU('U.1.(0).dat', 0);
m = size(U0,1);
n = size(U1,1);
%%
R00Test = padarray(R00Test, [m m]-size(R00Test),'post');
R01Test = padarray(R01Test, [m n]-size(R01Test),'post');
R11Test = padarray(R11Test, [n n]-size(R11Test),'post');
%%
nPositive = sum(R01Test(:)==1);
T = 7;
P_and_R = nan(T+1,2);
frobError = nan(T+1,1);
NPP = nan(T+1,1);
NTP = nan(T+1,1);
NP = nan(T+1,1);
acc = nan(T+1,1);
for t=0:T
  U0file = sprintf('U.0.(%d).dat', t);
  U1file = sprintf('U.1.(%d).dat', t);
  U0 = importU(U0file);
  U1 = importU(U1file);
  r = U0*U1';
  RHat = r;
  RHat(r<.75) = .5;
  RHat(r>=.75) = 1;
  iTest = R01Test ~= 0;
  RHat(~iTest) = 0;
  RHat = sparse(RHat);
  nPredPositive = sum(RHat(:)==1);
  TP = (R01Test==1) & (RHat==1);
  nTruePositive = sum(TP(:));
  Tr = TP | ((R01Test==.5)&(RHat==.5));
  
  prec = nTruePositive / nPredPositive;
  reca = nTruePositive / nPositive;
  P_and_R(t+1,:) = [prec reca];
  NPP(t+1) = nPredPositive;
  NTP(t+1) = nTruePositive;
  acc(t+1) = sum(Tr(:)) / sum(iTest(:));
end
%%
disp('for R01')
nPositive
NPP
NTP
P_and_R
acc
%%
rmse00 = nan(T+1,1);
for t=0:T
  U0file = sprintf('U.0.(%d).dat', t);
  U0 = importU(U0file);
  r = U0*U0';
  RHat = r;
  iTest = R00Test ~= 0;
  RHat(~iTest) = 0;
  RHat = sparse(RHat);
  er = RHat(:)-R00Test(:);
  rmse00(t+1) = sqrt(sum(er(:).^2)/sum(iTest(:)));
end
%%
disp('for R00')
rmse00

%%
rmse11 = nan(T+1,1);
for t=0:T
  U1file = sprintf('U.1.(%d).dat', t);
  U1 = importU(U1file);
  r = U1*U1';
  RHat = r;
  iTest = R11Test ~= 0;
  RHat(~iTest) = 0;
  RHat = sparse(RHat);
  er = RHat(:)-R11Test(:);
  rmse11(t+1) = sqrt(sum(er(:).^2)/sum(iTest(:)));
end
%%
disp('for R11')
rmse11