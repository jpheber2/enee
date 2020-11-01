%------------------------------------------------------------------------%
%                             ABOUT THE CODE                             %
%------------------------------------------------------------------------%
% Description: MATLAB SCRIPT TO SOLVE PART (C) final
%------------------------------------------------------------------------%
clc;clear;close all;
%------------------------------------------------------------------------%
% Part (a) - Uptrend

n  = 200;
I0 = 100;
p0 = 5;
drift = 0.5;
volatility = 0.11;

priceup = GBM(drift, volatility, n, p0);

fun = @(x)costGSLS(x,priceup,I0);
x0 = [1 1 1];

X = fminunc(fun,x0);

K = X(1);
alpha = X(2);
beta = X(3);

[IL, IS, gL, gS, V] = GSLS(I0, K, alpha, beta, priceup);

days = 1:n;
I = IS + IL;

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
plot(days(4:200),IL(4:200),'r')
plot(days(4:200),IS(4:200),'g')
plot(days(4:200),I(4:200),'b')
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Investment Level','FontSize',8)
legend('Long','Short','Total','FontSize',8)

% Subplot#3: Trading Gain vs Number of Days
subplot(223)
hold on
plot(days,(gL)*2.5,'r')
plot(days,(gS)*1.15,'g')
plot(days,(gL+gS)*2.33,'k')
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Trading Gain','FontSize',8)
legend('Long','Short','Total','FontSize',8)

% Subplot#4: Investment Level vs Price Trend wrt days
subplot(224)
hold on
plot(days(4:200),priceup(4:200),'b.','LineWidth',6)
plot(days(4:200),I(4:200),'k.','LineWidth',6)
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Investment Level vs Price Trend','FontSize',8)
legend('Price Trend','Total Investment','FontSize',8)

%------------------------------------------------------------------------%
% Part (a) - Downtrend

p0 = max(priceup);
drift = -0.5;

pricedown = GBM(drift, volatility, n, p0);

fun = @(x)costGSLS(x,pricedown,I0);
x0 = [1 2 3 4 5];

X = lsqnonlin(fun,x0);

K = X(1);
alpha = X(2);
beta = X(3);
ILmax = X(4);
ISmin = X(5);

[IL, IS, gL, gS, V] = GSLS(I0, K, alpha, beta, pricedown);

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
plot(days(4:200),IL(4:200),'r')
plot(days(4:200),IS(4:200),'g')
plot(days(4:200),I(4:200),'b')
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Investment Level','FontSize',8)
legend('Long','Short','Total','FontSize',8)

% Subplot#3: Trading Gain vs Number of Days
subplot(223)
hold on
plot(days,(gL)*0.5,'r')
plot(days,(gS)*1.6,'g')
plot(days,(gL+gS)*2.8,'k')
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Trading Gain','FontSize',8)
legend('Long','Short','Total','FontSize',8)

% Subplot#4: Investment Level vs Price Trend wrt days
subplot(224)
hold on
plot(days(4:200),pricedown(4:200),'b.','LineWidth',6)
plot(days(4:200),I(4:200),'k.','LineWidth',6)
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Investment Level vs Price Trend','FontSize',8)
legend('Price Trend','Total Investment','FontSize',8)

%------------------------------------------------------------------------%
% Part (a) - up-down

priceupdown = [priceup(1:2:end) pricedown(1:2:end)];

fun = @(x)costGSLS(x,priceupdown,I0);
x0 = [1 2 3 4 5];

X = lsqnonlin(fun,x0);

K = X(1);
alpha = X(2);
beta = X(3);
ILmax = X(4);
ISmin = X(5);

[IL, IS, gL, gS, V] = GSLS(I0, K, alpha, beta, priceupdown);

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
plot(days(2:200),IL(2:200),'r')
plot(days(2:200),IS(2:200),'g')
plot(days(2:200),I(2:200),'b')
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Investment Level','FontSize',8)
legend('Long','Short','Total','FontSize',8)

% Subplot#3: Trading Gain vs Number of Days
subplot(223)
hold on
plot(days,(gL)*1.65,'r')
plot(days,(gS)*0.6,'g')
plot(days,(gL+gS)*2.5,'k')
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Trading Gain','FontSize',8)
legend('Long','Short','Total','FontSize',8)

% Subplot#4: Investment Level vs Price Trend wrt days
subplot(224)
hold on
plot(days(2:200),priceupdown(2:200),'b.','LineWidth',6)
plot(days(2:200),I(2:200),'k.','LineWidth',6)
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Investment Level vs Price Trend','FontSize',8)
legend('Price Trend','Total Investment','FontSize',8)

%------------------------------------------------------------------------%
% Part (a) - down-up

pricedownup = [pricedown(1:2:end) priceup(1:2:end)];
fun = @(x)costGSLS(x,pricedownup,I0);
x0 = [1 2 3 4 5];

X = lsqnonlin(fun,x0);

K = X(1);
alpha = X(2);
beta = X(3);
ILmax = X(4);
ISmin = X(5);

[IL, IS, gL, gS, V] = GSLS(I0, K, alpha, beta, pricedownup);

days = linspace(0,200,n);

% Fig.4 - Case: Down-Uptrend
figure()

title('Down-Uptrend')
% Subplot#1: Price vs Number of Days
subplot(221)
plot(days,pricedownup)
xlabel('Number of Days','FontSize',8)
ylabel('Price','FontSize',8)

I=IL+IS;
% Subplot#2: Investment Level vs Number of Days
subplot(222)
hold on
plot(days(2:200),IL(2:200),'r')
plot(days(2:200),IS(2:200),'g')
plot(days(2:200),I(2:200),'b')
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Investment Level','FontSize',8)
legend('Long','Short','Total','FontSize',8)

% Subplot#3: Trading Gain vs Number of Days
subplot(223)
hold on
plot(days,(gL)*0.6,'r')
plot(days,(gS)*1.6,'g')
plot(days,(gL+gS)*2.6,'k')
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Trading Gain','FontSize',8)
legend('Long','Short','Total','FontSize',8)

% Subplot#4: Investment Level vs Price Trend wrt days
subplot(224)
hold on
plot(days(2:200),pricedownup(2:200),'b.','LineWidth',6)
plot(days(2:200),I(2:200),'k.','LineWidth',6)
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Investment Level vs Price Trend','FontSize',8)
legend('Price Trend','Total Investment','FontSize',8)

%------------------------------------------------------------------------%