%% Computer Exercise 1 
% Valuation of Derivatives
% Kajsa Hansson Willis & Victoria Lagerstedt

clc
clear
%% 3.1

K = 100;
T = 1; 
r = 0.05;
S_0 = 90;
sigma = 0.2;
N = 3; 


[EuropeanPrice] = calculateEuropeanCallOption(K,T,r,S_0,sigma,N);

%% 3.2

N = [1, 3, 10]; 

for n=1:length(N)
    [AmericanPutPrice(n),S_T] = calculateAmericanPutOption(K, T, r, S_0, sigma, N(n));
end 

for n=1:length(N)
    [AmericanCallPrice(n),S_T] = calculateAmericanCallOption(K, T, r, S_0, sigma, N(n));
end 

ForwardPrice = calculateForwardPrice(T, r, S_0);

LeftIneq  = S_0 - K;
ForwardValue = S_0 - K*exp(-r*T);

disp('Our prices at time t=0')

fprintf(' N   AmPut     AmCall    S_0 - K   Call-Put  Value of forward today   Forward\n');
fprintf('------------------------------------------------------------------------------\n');

for n=1:length(N)
    fprintf('%2d  %7.3f   %7.3f   %8.3f  %9.3f   %12.3f   %14.3f\n', ...
        N(n), AmericanPutPrice(n), AmericanCallPrice(n), ...
        LeftIneq, AmericanCallPrice(n)-AmericanPutPrice(n), ForwardValue, ForwardPrice);
end

%% 3.3


N = 1:500; 

for n=1:length(N)
    RNVFPrice(n) = RiskNeutralValuationFormula(K,T,r,S_0,sigma,N(n));
end 

BSPrice = CalculateBSCallPrice(S_0,K,r,T,0, sigma);

figure;
plot(N, RNVFPrice, 'o-'); hold on;
yline(BSPrice, 'r--','LineWidth',1.5);
xlabel('Number of periods N');
ylabel('European call option price');
title('Convergence of binomial RNVF to Black–Scholes');
legend('Binomial (RNVF)','Black–Scholes','Location','best');
grid on;


%% Functions

function [Pi, S_T] = calculateEuropeanCallOption(K, T, r, S_0, sigma, N)
    dt = T/N;
    discount = exp(-r * dt);
    u = exp(sigma*sqrt(dt)); 
    d = 1/u; 
    qu = (exp(r*dt) - d) / (u - d);

    k  = 0:N;
    S_T = S_0 .* (u.^k) .* (d.^(N-k));
    value = max(S_T - K, 0);

    for i=N:-1:1
        value = discount*(qu*value(2:end)+(1-qu)*value(1:end-1));
        disp(value)
    end
    
        Pi = value(1);
end 


function [Pi, S_T] = calculateAmericanPutOption(K, T, r, S_0, sigma, N)
    dt = T/N;
    discount = exp(-r * dt);
    u = exp(sigma*sqrt(dt)); 
    d = 1/u; 
    qu = (exp(r*dt) - d) / (u - d);

    k  = 0:N;
    S_T = S_0 .* (u.^k) .* (d.^(N-k));
    value = max(K-S_T, 0);

    for i=N:-1:1
        cont_value = discount*(qu*value(2:end)+(1-qu)*value(1:end-1));
        
        k = 0:(i-1);
        S = S_0 .* (u.^k) .* (d.^((i-1)-k));

        intr_value = max(K - S, 0);
        value = max(cont_value,intr_value);
    end
    
     Pi = value(1);
end 

function [Pi, S_T] = calculateAmericanCallOption(K, T, r, S_0, sigma, N) % same as European though
    dt = T/N;
    discount = exp(-r * dt);
    u = exp(sigma*sqrt(dt)); 
    d = 1/u; 
    qu = (exp(r*dt) - d) / (u - d);

    k  = 0:N;
    S_T = S_0 .* (u.^k) .* (d.^(N-k));
    value = max(S_T-K, 0);

    for i=N:-1:1
        cont_value = discount*(qu*value(2:end)+(1-qu)*value(1:end-1));
        
        k = 0:(i-1);
        S = S_0 .* (u.^k) .* (d.^((i-1)-k));

        intr_value = max(S-K, 0);
        value = max(cont_value,intr_value);
    end
    
     Pi = value(1);
end 

function [price] = calculateForwardPrice(T, r, S_0)
    price = S_0*exp(r*T);
end 

function [price] = RiskNeutralValuationFormula(K, T, r, S_0, sigma, N)
    delta = T/N;
    discount = exp(-r * T);
    u = exp(sigma*sqrt(delta)); 
    d = 1/u; 
    qu = (exp(r*delta)-d)/(u-d);
    k  = 0:N;
    S_T = S_0 .* (u.^k) .* (d.^(N-k));
    payoff = max(S_T - K, 0); 

    price = discount*sum( binopdf(k,N,qu).* payoff ); % <-- faster
    %price = discount*sum(factorial(N) ./ (factorial(k) .* factorial(N-k)) .* qu.^k .* (1-qu).^(N-k).* payoff );

end 

function [price] = CalculateBSCallPrice(S_0,K,r,T,t, sigma)
    d1 = 1/(sigma*sqrt(T-t))*(log(S_0/K)+(r+0.5*sigma^2)*(T-t));
    d2 = d1-sigma*sqrt(T-t);

    price = S_0*normcdf(d1)-exp(-r*(T-t))*K*normcdf(d2);
    
end 

 
