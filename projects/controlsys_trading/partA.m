%------------------------------------------------------------------------%
%                             ABOUT THE CODE                             %
%------------------------------------------------------------------------%
% Description: MATLAB SCRIPT TO SOLVE PART (A)
%------------------------------------------------------------------------%
clc;clear;close all;
%------------------------------------------------------------------------%
% Part (a) - Uptrend

n  = 200;
I0 = 100;
p0 = 5;
drift = 0.5;
volatility = 0.05;

priceup = GBM(drift, volatility, n, p0);    

fun = @(x)cost(x,priceup,I0);
x0 = 1;

x = lsqnonlin(fun,x0);

K = x;
[IL, IS, gL, gS, V] = SLS(I0, K, priceup);

days = 1:n;

% Fig.1 - Case: Uptrend
figure()

title('Uptrend')
% Subplot#1: Price vs Number of Days
subplot(221)
plot(days,priceup)
xlabel('Number of Days','FontSize',8)
ylabel('Price','FontSize',8)

% Subplot#2: Investment Level vs Number of Days
subplot(222)
hold on
plot(days,IL,'r')
plot(days,IS,'g')
plot(days,IL+IS,'b')
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Investment Level','FontSize',8)
legend('Long','Short','Total','FontSize',8)

% Subplot#3: Trading Gain vs Number of Days
subplot(223)
hold on
plot(days,gL,'r')
plot(days,gS,'g')
plot(days,gL+gS,'k')
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Trading Gain','FontSize',8)
legend('Long','Short','Total','FontSize',8)

% Subplot#4: Investment Level vs Price Trend wrt days
subplot(224)
hold on
plot(days,priceup,'b.','LineWidth',6)
plot(days,(IL+IS),'k.','LineWidth',6)
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Investment Level vs Price Trend','FontSize',8)
legend('Price Trend','Total Investment','FontSize',8)

%------------------------------------------------------------------------%
% Part (a) - Downtrend

p0 = max(priceup);
drift = -0.5;

pricedown = GBM(drift, volatility, n, p0);

fun = @(x)cost(x,pricedown,I0);
x0 = 1;

x = lsqnonlin(fun,x0);

K = x;
[IL, IS, gL, gS, V] = SLS(I0, K, pricedown);

days = 1:n;

% Fig.2 - Case: Downtrend
figure()

title('Downtrend')
% Subplot#1: Price vs Number of Days
subplot(221)
plot(days,pricedown)
xlabel('Number of Days','FontSize',8)
ylabel('Price','FontSize',8)

% Subplot#2: Investment Level vs Number of Days
subplot(222)
hold on
plot(days,IL,'r')
plot(days,IS,'g')
plot(days,IL+IS,'b')
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Investment Level','FontSize',8)
legend('Long','Short','Total','FontSize',8)

% Subplot#3: Trading Gain vs Number of Days
subplot(223)
hold on
plot(days,gL,'r')
plot(days,gS,'g')
plot(days,gL+gS,'k')
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Trading Gain','FontSize',8)
legend('Long','Short','Total','FontSize',8)

% Subplot#4: Investment Level vs Price Trend wrt days
subplot(224)
hold on
plot(days,pricedown,'b.','LineWidth',6)
plot(days,(IL+IS),'k.','LineWidth',6)
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Investment Level vs Price Trend','FontSize',8)
legend('Price Trend','Total Investment','FontSize',8)

%------------------------------------------------------------------------%
% Part (a) - up-down

priceupdown = [priceup(1:2:end) pricedown(1:2:end)];

fun = @(x)cost(x,priceupdown,I0);
x0 = 1;

x = lsqnonlin(fun,x0);

K = x;
[IL, IS, gL, gS, V] = SLS(I0, K, priceupdown);

days = linspace(0,200,n);

% Fig.3 - Case: Up-Downtrend
figure()

title('Up-Downtrend')
% Subplot#1: Price vs Number of Days
subplot(221)
plot(days,priceupdown)
xlabel('Number of Days','FontSize',8)
ylabel('Price','FontSize',8)

% Subplot#2: Investment Level vs Number of Days
subplot(222)
hold on
plot(days,IL,'r')
plot(days,IS,'g')
plot(days,IL+IS,'b')
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Investment Level','FontSize',8)
legend('Long','Short','Total','FontSize',8)

% Subplot#3: Trading Gain vs Number of Days
subplot(223)
hold on
plot(days,gL,'r')
plot(days,gS,'g')
plot(days,gL+gS,'k')
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Trading Gain','FontSize',8)
legend('Long','Short','Total','FontSize',8)

% Subplot#4: Investment Level vs Price Trend wrt days
subplot(224)
hold on
plot(days,priceupdown,'b.','LineWidth',6)
plot(days,(IL+IS),'k.','LineWidth',6)
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Investment Level vs Price Trend','FontSize',8)
legend('Price Trend','Total Investment','FontSize',8)

%------------------------------------------------------------------------%
% Part (a) - down-up

pricedownup = [pricedown(1:2:end) priceup(1:2:end)];
fun = @(x)cost(x,pricedownup,I0);
x0 = 1;

x = lsqnonlin(fun,x0);

K = x;
[IL, IS, gL, gS, V] = SLS(I0, K, pricedownup);

days = linspace(0,200,n);

% Fig.4 - Case: Down-Uptrend
figure()

title('Down-Uptrend')
% Subplot#1: Price vs Number of Days
subplot(221)
plot(days,pricedownup)
xlabel('Number of Days','FontSize',8)
ylabel('Price','FontSize',8)

% Subplot#2: Investment Level vs Number of Days
subplot(222)
hold on
plot(days,IL,'r')
plot(days,IS,'g')
plot(days,IL+IS,'b')
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Investment Level','FontSize',8)
legend('Long','Short','Total','FontSize',8)

% Subplot#3: Trading Gain vs Number of Days
subplot(223)
hold on
plot(days,gL,'r')
plot(days,gS,'g')
plot(days,gL+gS,'k')
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Trading Gain','FontSize',8)
legend('Long','Short','Total','FontSize',8)

% Subplot#4: Investment Level vs Price Trend wrt days
subplot(224)
hold on
plot(days,pricedownup,'b.','LineWidth',6)
plot(days,(IL+IS),'k.','LineWidth',6)
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Investment Level vs Price Trend','FontSize',8)
legend('Price Trend','Total Investment','FontSize',8)

%------------------------------------------------------------------------%