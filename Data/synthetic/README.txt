3 entities
1000 observations of each entity
100 features for each entity - 50 Bernoulli, 50 normal

latent dimension 10
standard deviation 1 for all normal distributions

all ratings are positive

R.k.l.train - 50,000 sparse entries of relation matrix between entities k and l
R.k.l.test - non-sparse relation matrix between entities k and l;
has exactly the entries missing from the training data; 
rating of 0 indicates that the entry was used for training

F.normal.k - non-sparse normal feature matrix (n x d.normal) for entity k
F.bernoulli.k - sparse Bernoulli feature matrix (n x d.normal) for entity k