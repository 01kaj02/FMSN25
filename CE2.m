%% Computer Exercise 2
% Valuation of Derivatives
% Kajsa Hansson Willis & Victoria Lagerstedt

clc
clear
%% 4.1 European call option


K = 80;
T = 1; 
r = 0.01; 
S_0 = 90;
sigma = 0.6; 

N = 10000;


% Crude Monte Carlo

mu = (log(S_0) + (r - 0.5*sigma^2)*T);

X = mvnrnd(mu, sigma^2*T, N);  
ST = exp(X);   
Phi = max(ST - K, 0);
price_crude = exp(-r*T) * mean(Phi,1);
variance = var(Phi,0,1) / N;
stderr_crude = sqrt(variance);

disp('Crude price:')
disp(price_crude);

% Antithetic variates

Nat = N/2;

Z = mvnrnd(0,1,Nat); 
Zat = -Z;  

ST_at1 = exp(mu+sigma*sqrt(T)*Z);
ST_at2 = exp(mu+sigma*sqrt(T)*Zat);

Phi1 = max(ST_at1 - K, 0);
Phi2 = max(ST_at2 - K, 0);

Phi_at = 0.5*(Phi1 + Phi2);
price_at = exp(-r*T) * mean(Phi_at,1);

variance_at = var(Phi_at) / Nat;
stderr_at = sqrt(variance_at);

disp('Antithetic price:')
disp(price_at);


%Control variates ST

Z = mvnrnd(0,1,N); 
X = S_0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z);
ST = X;

payoff = exp(-r*T)*max(X - K, 0);

EST = S_0 * exp(r*T);

b = cov(payoff, ST);
b = b(1,2) / var(ST);  

Y_CV = payoff - b*(ST-EST);

price_CV = mean(Y_CV);

variance = var(Y_CV)/N;

stderr_cv = sqrt(variance);

disp('Control Variate price:')
disp(price_CV);


% Black and Scholes
 sigmaT = sigma*sqrt(T);
 d1 = (log(S_0/K) + (r + 0.5*sigma^2)*T) / sigmaT;
 d2 = d1 - sigmaT;
 price_BS = S_0*normcdf(d1) - K*exp(-r*T)*normcdf(d2);

disp('Black-Scholes price:')
disp(price_BS);

%% 4.2 Basket call option

K = [80, 100, 120];

T = 4; 
n = 12; 
r = 0.02; 
S_0 = 100; 

sigma = 0.4;
corr = 0.6;
Sigma = sigma^2*((1-corr)*eye(n) + corr*ones(n)); 
Sigma = Sigma*T;


N_array = [1000, 10000, 100000];
prices = zeros(length(N_array), length(K));

mu = (log(S_0) + (r - 0.5*sigma^2)*T) * ones(1,n);


for i = 1:length(N_array)
    N = N_array(i);

    X = mvnrnd(mu, Sigma, N);  
    ST = exp(X);   

    basket = mean(ST,2); % as they are equally weighted
        
    Phi = max(basket - K, 0);

    prices(i,:) = exp(-r*T) * mean(Phi,1);

end 
Tprices  = array2table(prices, 'VariableNames', {'K80','K100','K120'}, ...
                                'RowNames', compose('N=%g',N_array));

disp('Arithmetic Basket Call — Monte Carlo Prices:');
disp(Tprices);


%% 4.3 Stochastic volatility


kappa=10; 
theta=0.16; 
sigma=0.1; 
rho=-0.8; 
V_0=0.16;
S_0=90; 
K=80; 
r=0.01; 
T=1;
N=10000; 
M=100; 
h=T/M; 
disc=exp(-r*T);

S = zeros(N,1);

for i=1:N
    S_t=S_0; 
    V_t=V_0;
    for k=1:M
        G1=randn; 
        G2=randn;            
        dWv = sqrt(h)*G1;               % drives V
        dWs = sqrt(h)*(rho*G1 + sqrt(1-rho^2)*G2); % correlated for S

        Vpos = max(V_t,0);

        S_t  = S_t * exp((r-0.5*Vpos)*h + sqrt(Vpos)*dWs);

        V_t  = V_t + kappa*(theta - V_t)*h + sigma*sqrt(Vpos)*dWv + 0.25*sigma^2*h*(G1^2 - 1);
        V_t  = max(V_t,0);
       
    end
    S(i)=S_t;
end

payoff = disc*max(S-K,0);
price_MC = mean(payoff);
stderr   = std(payoff)/sqrt(N);


% Fourier inversion technique 
par = [V_0 kappa theta sigma rho];
price_fit = opt_price('Heston',par,1,1,K,S_0,r,T); 


fprintf('Prices with Monte Carlo simulation: %.4f Standard error: %.4f \n', price_MC, stderr);

fprintf('Prices with Fourier Inversion Technique: %.4f \n', price_fit);



% Price of up-out contract 

K = 40;
B = 100;
S_0 = 50;
disp(opt_price('Heston',par,1,1,K,S_0,r,T));
Suo = zeros(N,1);


for i=1:N
    S_t=S_0; 
    V_t=V_0;
    for k=1:M
        G1=randn; 
        G2=randn;            
        dWv = sqrt(h)*G1;               % drives V
        dWs = sqrt(h)*(rho*G1 + sqrt(1-rho^2)*G2); % correlated for S

        Vpos = max(V_t,0);

        S_t  = S_t * exp((r-0.5*Vpos)*h + sqrt(Vpos)*dWs);

        V_t  = V_t + kappa*(theta - V_t)*h + sigma*sqrt(Vpos)*dWv + 0.25*sigma^2*h*(G1^2 - 1);
        V_t  = max(V_t,0);

         if (S_t >= B)
            S_t = 0;
            break
        end 
    end
    Suo(i)=S_t;
end

payoff = disc*max(Suo-K,0);
price_suo = mean(payoff);
stderr_suo   = std(payoff)/sqrt(N);
fprintf('MC price of up-and-out option: %.4f Standard error: %.4f \n', price_suo, stderr_suo);


%%

kappa = 12;
par = [V_0 kappa theta sigma rho];

price_fit = opt_price('Heston',par,1,1,K,S_0,r,T); 

