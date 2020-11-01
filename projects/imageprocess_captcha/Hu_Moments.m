function inv_moments = Hu_Moments(eta)

%Calculation of various invariant Hu's moments
inv_moments(1) = 10^4*(eta(3,1) + eta(1,3));
inv_moments(2) = 10^4*((eta(3,1) - eta(1,3))^2 + (4*eta(2,2)^2));
inv_moments(3) = 10^4*((eta(4,1) - 3*eta(2,3))^2 + (3*eta(3,2) - eta(1,4))^2);
inv_moments(4) = 10^4*((eta(4,1) + eta(2,3))^2 + (eta(3,1) + eta(1,4))^2);
inv_moments(5) = 10^4*((eta(4,1) - 3*eta(2,3))*(eta(4,1) + eta(2,3))*((eta(4,1) + eta(2,3))^2 - 3*((eta(3,2) + eta(1,4))^2)) + (3*(eta(3,2) - eta(1,4)))*(eta(3,2) + eta(1,4))*(3*(eta(4,1) + eta(2,3))^2 - (eta(3,2) + eta(1,4))^2));
inv_moments(6) = 10^4*((eta(3,1) - eta(1,3))*((eta(4,1)+eta(2,3))^2 - (eta(3,2)+ eta(1,4))^2) + 4*eta(2,2)*((eta(4,1) + eta(2,3))*(eta(3,2) + eta(1,4))));
inv_moments(7) = 10^4*((3*eta(3,2) - eta(1,4))*(eta(4,1) + eta(2,3))*((eta(4,1) + eta(2,3))^2 - 3*(eta(3,2)-eta(1,4))^2) - (eta(4,1) - 3*eta(2,3))*(eta(3,2) + eta(1,4))*(3*(eta(4,1) + eta(2,3))^2 - (eta(3,2) + eta(1,4))^2));