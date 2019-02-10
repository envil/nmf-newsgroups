%% Matrix Decompositions in Data Analysis
% Winter 2019
% Assignment file
% FILL IN THE FOLLOWING INFORMATION
% Name: 
% Student ID:
%

%% Main function
% In Matlab, we can't mix script files and function files, so we'll have 
% a main function called assignment that will call the other functions
% that can be added to this file.
%
% For prototyping, it's better to call the functions from Matlab command line, 
% as the functions don't save their working space. 

function assignment() 
  % To read the data, we can use the readtable command
  T = readtable('news.csv');
  A = T.Variables;
  terms = T.Properties.VariableNames;

  %% Task 1
  % FILL IN HERE the code for doing Task 1
  % One round of nmf_als would be like
  [W, H, errs] = nmf_als(A, 20);
  % To plot the errors per iterations, use
  figure, plot(errs);
  title('Convergence of NMF ALS');
  xlabel('Iteration');
  ylabel('||A - WH||_F^2');
  % DO THE OTHER NMF methods similarly and add here code to call them
  % and to do the comparisons. 

  %% Task 2
  % We use B for normalised A
  B = A./sum(sum(A));
  % DO THE NMF

  % If H is an output of some NMF algorihtm, we get the top-10 entries
  % of the first row of H as follows
  h = H(1,:);
  [~, I] = sort(h, 'descend');
  for i=1:10
    fprintf('%s\t%f\n', terms{I(i)}, h(I(i)));
  end

  %% Task 3
  % To compute K-L, we need the z-scores
  Z = normalize(A); % This requires MATLAB 2018a
  [U, S, V] = svds(Z, 20);
  KL = Z*V; % N.B. svds returns correct-sized V
  %% COMPUTE pLSA
  % To compute k-means, use kmeans; Replicates controls the number of 
  % restarts. 
  [idx] = kmeans(KL, 20, 'Replicates', 20);
  % DO K-MEANS for other cases 
  % To calculate the NMI, use
  nmi_KL = nmi_news(idx);
  fprintf('NMI (KL) = %f\n', nmi_KL);
  % DO NMI for other clusterings
end


function [W, H, errs] = nmf_als(A, k, maxiter) 
% Boilerplate function for NMF with ALS
% Errs contains the errors per iteration
% Use 300 iterations if no iters is given
  if nargin < 3 
    maxiter = 300; 
  end
  % Take dimensions and init W
  [n, m] = size(A);
  W = rand(n, k);
  H = rand(k, m); % This is actually not needed
  % Init errs to NaNs
  errs = nan(maxiter, 1);
  for i = 1:maxiter,
    % FILL IN an update for H and W
    errs(i) = norm(A - W*H, 'fro')^2; % Squared Frobenius
  end

end


function z = nmi(x, y)
% Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)) of two discrete variables x and y.
% Input:
%   x, y: two integer vector of the same length 
% Ouput:
%   z: normalized mutual information z=I(x,y)/sqrt(H(x)*H(y))
% Written by Mo Chen (sth4nth@gmail.com).
% https://de.mathworks.com/matlabcentral/fileexchange/29047-normalized-mutual-information
  assert(numel(x) == numel(y));
  n = numel(x);
  x = reshape(x,1,n);
  y = reshape(y,1,n);
  l = min(min(x),min(y));
  x = x-l+1;
  y = y-l+1;
  k = max(max(x),max(y));
  idx = 1:n;
  Mx = sparse(idx,x,1,n,k,n);
  My = sparse(idx,y,1,n,k,n);
  Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
  Hxy = -dot(Pxy,log2(Pxy));
  % hacking, to elimative the 0log0 issue
  Px = nonzeros(mean(Mx,1));
  Py = nonzeros(mean(My,1));
  % entropy of Py and Px
  Hx = -dot(Px,log2(Px));
  Hy = -dot(Py,log2(Py));
  % mutual information
  MI = Hx + Hy - Hxy;
  % normalized mutual information
  z = sqrt((MI/Hx)*(MI/Hy));
  z = max(0,z);

end

function z = nmi_news(x)
% Computes the NMI between the news ground truth and given clustering
  gt = load('news_ground_truth.txt');
  z = nmi(x, gt);
end 

