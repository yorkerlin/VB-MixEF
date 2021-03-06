function [y, X, y_te, X_te] = get_data_log_reg(name, seed)
%name: name of a dataset
%seed: seed used for creating a train/test set
%y (y_te): labels for a train (test) set
%X (X_te): features for a train (test) set
%Note: for classification problems, we use 0-1 encoding in labels.
switch name
case 'synth'
  setSeed(seed);
  N = 5000;
  D = 2;
  s2 = .01;
  X = randn(N,D);
  D = D + 1;
  w = [0.1; -1; +1];
  eta = [ones(N,1) X]*w + s2*randn(N,1);
  y = sign(eta);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);

case {'a2a','a3a','a4a','a5a','a6a','a7a'}
  load(name);
  [N,D] = size(X);
  X = [ones(N,1) X];
  [N_te,D] = size(X_te);
  X_te = [ones(N_te,1) X_te];
  y = (y+1)/2;
  y_te = (y_te+1)/2;
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)

case {'svmguide3'}
  load(name);
  [N,D] = size(X);
  X = [ones(N,1) X];
  [N_te,D] = size(X_te);
  X_te = [ones(N_te,1) X_te];
  y = (y+1)/2;
  y_te = (y_te+1)/2;
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)

case 'svmguide1'
  load(name);
  [N,D] = size(X);
  X = [ones(N,1) X];
  [N_te,D] = size(X_te);
  X_te = [ones(N_te,1) X_te];
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)

case 'a1a'
  load(name);
  [N,D] = size(X);
  X = [ones(N,1) X zeros(N,4)];
  [N_te,D] = size(X_te);
  X_te = [ones(N_te,1) X_te];
  y = (y+1)/2;
  y_te = (y_te+1)/2;
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)

case {'colon-cancer'}
  load(name);
  [N,D] = size(X);
  X = [ones(N,1) X];
  y = (y+1)/2;
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);

case {'duke'}
  load(name);
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  y = (y+1)/2;
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);

case {'leukemia'}
  load(name);
  [N,D] = size(X);
  X = [ones(N,1) X];
  [N_te,D] = size(X_te);
  X_te = [ones(N_te,1) X_te];
  y = (y+1)/2;
  y_te = (y_te+1)/2;
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)

case {'gisette_scale'}
  load(name);
  X = [X; X_te];
  y = [y; y_te];
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  y = (y+1)/2;
  unique(y)
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);

case {'covtype_binary_scale'}
  load(name);
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  y = y-1;
  setSeed(seed);
  %[X, y, X_te, y_te] = split_data(y, X, 0.5);
  [X, y, X_te, y_te] = split_data(y, X, 0.8);
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)

case {'SUSY'}
  load('SUSY.amat','-mat');
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.9);
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)

case {'australian_scale'}
  load(name);
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  y = (y+1)/2;
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)

case {'breast_cancer_scale'}
  load(name);
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  y = (y-2)/2;
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)


case 'ionosphere'
  data = csvread('ionosphere.csv');
  X = data(:,1:end-1); 
  y = data(:,end);
  y = (y+1)/2;
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);
  clear data;
 
case 'sonar' 
  data = csvread('sonar.csv');
  X = data(:,1:end-1); 
  y = data(:,end);
  y = (y+1)/2;
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);
  clear data;
 

case 'diabetes_scale'
  data = csvread('diabetes_scale.csv');
  X = data(:,1:end-1); 
  y = data(:,end);
  y = (y+1)/2;
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);
  clear data;
 
case 'german.numer_scale'
  data = csvread('german.numer_scale.csv');
  X = data(:,1:end-1); 
  y = data(:,end);
  y = (y+1)/2;
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);
  clear data;
  
case 'heart_scale'
  data = csvread('heart_scale.csv');
  X = data(:,1:end-1); 
  y = data(:,end);
  y = (y+1)/2;
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);
  clear data;
 
case 'skin_nonskin'
  data = csvread('skin_nonskin.csv');
  X = data(:,1:end-1); 
  y = data(:,end);
  y = (y-1);
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);
  clear data;
 
  
case 'splice.all'
  data = csvread('splice.all.csv');
  X = data(:,1:end-1); 
  y = data(:,end);
  y = (y+1)/2;
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);
  clear data;

 case 'german.numer'
  data = csvread('german.numer.csv');
  X = data(:,1:end-1); 
  y = data(:,end);
  y = (y+1)/2;
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);
  clear data;
  
case 'heart'
  data = csvread('heart.csv');
  X = data(:,1:end-1); 
  y = data(:,end);
  y = (y+1)/2;
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);
  clear data;
 

case 'diabetes'
  data = csvread('diabetes.csv');
  X = data(:,1:end-1); 
  y = data(:,end);
  y = (y+1)/2;
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);
  clear data;
 
case 'bank'
  data = csvread('bank.csv');
  X = data(:,1:end-1); 
  y = data(:,end);
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);
  clear data;
 
case 'breast-cancer'
  data = csvread('breast-cancer.csv');
  X = data(:,1:end-1); 
  y = data(:,end);
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  y = (y-2)/2;
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)

case 'phishing'
  data = csvread('phishing.csv');
  X = data(:,1:end-1); 
  y = data(:,end);
  [N,D] = size(X);
  X = [ones(N,1) full(X)];
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);
  assert ( length(unique(abs(y))) == 2)
  assert ( length(unique(abs(2*y-1))) == 1)



otherwise
  error('no such name');
end
end

function [XTr, yTr, XTe, yTe] = split_data(y, X, prop)

  N = size(y,1);
	idx = randperm(N);
  Ntr = floor(prop * N);
	idxTr = idx(1:Ntr);
	idxTe = idx(Ntr+1:end);
  XTr = X(idxTr,:);
  yTr = y(idxTr);
  XTe = X(idxTe,:);
  yTe = y(idxTe);
end

