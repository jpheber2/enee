function err = costGSLS(X,price,I0)

K = X(1);
alpha = X(2);
beta = X(3);

[IL, IS, gL, gS, V] = GSLS(I0, K, alpha, beta, price);
err = norm((V)-price*1.65)-var(price);

end