function [ Z ] = YtoZ( Y )
%YtoZ convert Y bus martix to Z bus matrix
%   converts Y matrix to Z by use of LU decomposition and row by row
%   substituition, rather than inversion

[m n]=size(Y);
if (m ~= n )
disp ( 'Matrix must be square' );
return;
end;

U = Y; 
L = eye(n,n); 

for j = 1:n-1        
	for i = j+1:n  x)1

I = eye(size(Y));
Z = zeros(size(Y));

for i = 1:n
j = I(:,i);
		Z(:,i) = U\ (L\ j);
end

end

