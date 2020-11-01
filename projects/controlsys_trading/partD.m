fun = @(x)costGSLS(x,price,I0);
x0 = [1 1 1];
%fminunc
X = lsqnonlin(fun,x0);

K = X(1);
alpha = X(2);
beta = X(3);

[IL, IS, gL, gS, V] = GSLS(I0, K, alpha, beta, price);

days = 1:n;
I = IS + IL;

% Fig.1 - Case: Uptrend
figure()

title('Uptrend')
% Subplot#1: Price vs Number of Days
subplot(221)
plot(days,price*1.1)
xlabel('Number of Days','FontSize',8)
ylabel('Price','FontSize',8)

% Subplot#2:  Investment Level vs Number of Days
subplot(222)
hold on
plot(days,-(IS-15)*0.7,'r')
plot(days,-(IL-15)*0.7,'g')
plot(days,-(I-15)*0.7,'b')
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Investment Level','FontSize',8)
legend('Long','Short','Total','FontSize',8)

% Subplot#3: GSLS Gain vs Number of Days
subplot(223)
hold on
plot(days,(gL),'r')
plot(days,(gS),'g')
plot(days,smooth(gL+gS),'k')
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Trading Gain','FontSize',8)
legend('Long','Short','Total','FontSize',8)

% Subplot#4: Investment Level vs Price Trend wrt days
I=smooth(I);
subplot(224)
hold on
plot(days,price*1.1-20,'b.','LineWidth',6)
plot(days,-(I-15)*0.7,'k.','LineWidth',6)
hold off
xlabel('Number of Days','FontSize',8)
ylabel('Investment Level vs Price Trend','FontSize',8)
legend('Price Trend','Total Investment','FontSize',8)