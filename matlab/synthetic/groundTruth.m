clear
close all
home
load synthetic
%%
rmse = nan(nEntities);
for i=1:nEntities
  for j=i+1:nEntities
    ER = U{i}*U{j}';
    RTest = testR{i,j};
    ER(RTest==0)=0;
    er = ER(:)-RTest(:);
    rmse(i,j) = sqrt(sum(er.^2)/sum(RTest(:)~=0));
  end
end
rmse