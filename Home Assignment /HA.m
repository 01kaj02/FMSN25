%% Home Assignment
%  FMSN25 VALUATION OF DERIVATIVE ASSETS
%  Kajsa Hansson Willis

clc;
clear;

%% Part B 

%% 1

n = 12; 
r = 0.02;
s_0 = 100;
c_i = (1/n)*ones(n,1); 

K = [80, 100, 120];
T = 4; 
C_lower = zeros(size(K)); 

sigma = 0.4;
rho = 0.6;

Sigma = sigma^2*((1-rho)*eye(n) + rho*ones(n));

mu_GB = sum(c_i .* (log(s_0) + (r - 0.5*sigma.^2)*T));

var_GB = T * (c_i' * Sigma * c_i);   
sigZ = sqrt(var_GB);

EG = exp(mu_GB + 0.5 * var_GB);


 for j = 1:length(K)
     d1 = (mu_GB - log(K(j)) + var_GB) / sigZ;
     d2 = d1 - sigZ;

   C_lower(j) = exp(-r*T) * (EG * normcdf(d1) - K(j) * normcdf(d2));
 end 

 disp(table(K.', C_lower.', 'VariableNames', {'K','Lower Bound'}))


%% 2 Upper Bounds

sigmaT = sigma*sqrt(T);
c_i = (1/n)*ones(n,1); 

C_upper = zeros(size(K)); 

for j = 1:length(K)
    sum_i = 0;
    for i = 1:n
        d1 = (log(s_0/K(j)) + (r + 0.5*sigma^2)*T) / sigmaT;
        d2 = d1 - sigmaT;
        call_i = s_0*normcdf(d1) - K(j)*exp(-r*T)*normcdf(d2);
        sum_i = sum_i + c_i(i)*call_i;  
    end
    C_upper(j) = sum_i;
end

disp(table(K.', C_upper.', 'VariableNames', {'K','Upper Bound'}))

%% 3 Monte Carlo price of arithmetic basket call

N_array = [1000, 10000, 100000];
T = 4; 

%% Monte Carlo
SigmaxT = Sigma*T;  

prices = zeros(length(N_array), length(K));
stderr = zeros(length(N_array), length(K));

mu = (log(s_0) + (r - 0.5*sigma^2)*T) * ones(1,n);

for i = 1:length(N_array)
    N = N_array(i);

    X = mvnrnd(mu, SigmaxT, N);  
    ST = exp(X);   
    basket = mean(ST,2); % only as they are equally weighted
        
    Phi = max(basket - K, 0);

    prices(i,:) = exp(-r*T) * mean(Phi,1);
    variance   = exp(-2*r*T)*var(Phi,0,1) / N;

    stderr(i,:) = sqrt(variance);

end 
Tprices  = array2table(prices, 'VariableNames', {'K80','K100','K120'}, ...
                                'RowNames', compose('N=%g',N_array));

Tstderr  = array2table(stderr, 'VariableNames', {'K80','K100','K120'}, ...
                                'RowNames', compose('N=%g',N_array));

disp('Arithmetic Basket Call – Monte Carlo Prices:');
disp(Tprices);

disp('Standard Errors crude Monte Carlo:');
disp(Tstderr);


%% Control variate 

prices_CV = zeros(length(N_array), length(K));
stderr_CV = zeros(length(N_array), length(K));

for i = 1:length(N_array)
    N = N_array(i);

    X = mvnrnd(mu, SigmaxT, N);  
    ST = exp(X);

    basketA = mean(ST,2);      
    basketG = geomean(ST,2);   

    for j = 1:length(K)
        Y = exp(-r*T) * max(basketA - K(j), 0); 
        Xg = exp(-r*T) * max(basketG - K(j), 0); 

        EX = C_lower(j); % from B1

        b = cov(Y, Xg);
        b = b(1,2) / var(Xg);

        Y_CV = Y - b * (Xg - EX);

        prices_CV(i,j) = mean(Y_CV);
        stderr_CV(i,j) = std(Y_CV) / sqrt(N);
    end
end

Tprices_CV = array2table(prices_CV, 'VariableNames', {'K80','K100','K120'}, ...
                                      'RowNames', compose('N=%g',N_array));
Tstderr_CV = array2table(stderr_CV, 'VariableNames', {'K80','K100','K120'}, ...
                                      'RowNames', compose('N=%g',N_array));

disp('Arithmetic Basket Call – Control Variate Prices:');
disp(Tprices_CV);

disp('Control Variate Standard Errors:');
disp(Tstderr_CV);


%% 4
clear
load('HA25_data.mat');

NA = 100; 
r = -0.001;
N = 10000;
n = length(S0);


%% a
T = 5;
factor = 1.1;

Sigma = diag(sigma) * rho * diag(sigma);

SigmaxT = Sigma*T;  

mu = (log(S0) + (r - 0.5*sigma.^2)*T);

X = mvnrnd(mu, SigmaxT, N);  
ST = exp(X);   

basketRel = (ST ./ S0) * c;  

pos =  mean(max(basketRel - 1, 0)); 

pr = (factor*exp(r*T) - 1) / pos;


price0  = exp(-r*T) * NA * (1 + pr*pos);

fprintf('pr = %.6f\n', pr);
fprintf('Price = %.4f (target %.4f)\n', price0, factor*NA);



%% b

T = 4; 

SigmaxT4 = Sigma*T;  

mu4 = (log(S1) + (r - 0.5*sigma.^2)*T);

X = mvnrnd(mu4, SigmaxT4, N);  
ST4 = exp(X);   

basketRel = (ST4 ./ S0) * c;
payoff = NA * (1 + pr * max(basketRel - 1, 0));

expPayoff = mean(payoff);

price = exp(-r*T) * expPayoff;

fprintf('Price = %.4f \n', price);


