Machine = xlsread('/Users/Madeleine/Downloads/six_bus_system.xlsx',1);

Line = xlsread('/Users/Madeleine/Downloads/six_bus_system.xlsx',2);

num_bus = max( max(Line(:,1)), max(Line(:,2)) );

num_line = length(Line(:,1));

num_mc = length(Machine(:,1));

Y_line = zeros(num_bus,num_bus);
for k=1:num_line
admittance=1./(Line(k,3)+i*Line(k,4));
Y_line(Line(k,1),Line(k,2))=admittance;
Y_line(Line(k,2),Line(k,1))=admittance;
end

Y_shunt = zeros(num_bus,num_bus);
for k = 1:num_line
shuntadmittance = (Line(k,5)+i*Line(k,6)); 
Y_shunt(Line(k,1),Line(k,2)) = shuntadmittance;
Y_shunt(Line(k,2),Line(k,1)) = shuntadmittance;
end
Y_mach = zeros(num_bus,1);
for k = 1:num_mc
Y_mach(Machine(k,1)) = 1./( i*Machine(k,4));
end

Y_tl = zeros(num_bus,num_bus);
Y_mc = zeros(num_bus,num_bus);

for k = 1:num_bus
for j = k:num_bus
%Fill off-diagonal elements
if (k ~= j)
Y_tl(k,j) = -1*Y_line(k,j);
Y_tl(j,k) = Y_tl(k,j);
end

%Fill diagonal elements
if (k == j)
% Find the sum of all admittances connected to Bus i 
Y_tl(k,j) = sum(Y_line(k,:))+(sum(Y_shunt(k,:))/2);
end 
end
end

% Fill Ybus
for k = 1:num_bus
for j = k:num_bus
%Fill off-diagonal elements
if (k ~= j)
Y_mc(k,j) = -1*Y_line(k,j);
Y_mc(j,k) = Y_mc(k,j);
end
%Fill diagonal elements
if (k == j)
% Find the sum of all admittances connected to Bus i             
Y_mc(k,j) = sum(Y_line(k,:))+(sum(Y_shunt(k,:))/2)+Y_mach(k,:);
end 
end
end

disp('Y bus - transmission lines only');
disp(Y_tl);
disp('Y bus - equipment impedance included');
disp(Y_mc)