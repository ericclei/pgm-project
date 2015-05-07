clear
home
close all
%%
[i,j,v] = import_data_test('data_test.csv');
v(v == 0) = .5;
R01Test = sparse(i,j,v);
%%
[i,j,v] = import_data_test('gene_gene_test.csv');
n = max(i);
R00Test = sparse(n,n);
R00Test(sub2ind([n,n],i,j)) = v;
%%
[i,j,v] = import_data_test('disease_disease_test.csv');
n = max(i);
R11Test = sparse(n,n);
R11Test(sub2ind([n,n],i,j)) = v;
%%
save RTest R*Test