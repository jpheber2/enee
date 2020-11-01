function err = cost(K,price,I0)

[IL, IS, gL, gS, V] = SLS(I0, K, price);

err = norm((V)-price);

end