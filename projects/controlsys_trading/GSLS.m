function [IL, IS, gL, gS, V] = GSLS(I0, K, alpha, beta, price)
IL(1) = I0;
IS(1) = -alpha*I0;
gL(1) = 0;
gS(1) = 0;
V(1) = 0;
n = length(price);
for i=2:n
rho = (price(i)-price(i-1))/price(i);
gL(i) = gL(i-1) + rho*IL(i-1);
gS(i) = gS(i-1) + rho*IS(i-1);
IL(i) = IL(1) + K*gL(i);
IS(i) = alpha*IS(1) - beta*K*gS(i);
V(i) = gS(i) + gL(i);
end
end