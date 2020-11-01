function price = GBM(drift, volatility, n, p0)
price(1) = p0;
for i=2:n
price(i) = (1+drift/n+volatility/sqrt(n)*randn)*price(i-1);
end
end