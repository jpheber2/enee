{\rtf1\ansi\ansicpg1252\cocoartf1504\cocoasubrtf830
{\fonttbl\f0\fmodern\fcharset0 Courier;\f1\froman\fcharset0 Times-Roman;\f2\fmodern\fcharset0 CourierNewPSMT;
}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sl280\partightenfactor0

\f0\fs24 \cf2 \expnd0\expndtw0\kerning0
Machine = xlsread('/Users/Madeleine/Downloads/six_bus_system.xlsx',1);\
\
Line = xlsread('/Users/Madeleine/Downloads/six_bus_system.xlsx',2);\
\
num_bus = max( max(Line(:,1)), max(Line(:,2)) );\
\
num_line = length(Line(:,1));\
\
num_mc = length(Machine(:,1));\
\
Y_line = zeros(num_bus,num_bus);
\f1 \cf2 \
\pard\pardeftab720\sl320\partightenfactor0

\f2\fs26\fsmilli13333 \cf2 for k=1:num_line\
admittance=1./(Line(k,3)+i*Line(k,4));\
Y_line(Line(k,1),Line(k,2))=admittance;\
Y_line(Line(k,2),Line(k,1))=admittance;\
end\
\
\pard\pardeftab720\sl280\partightenfactor0

\f0\fs24 \cf2 Y_shunt = zeros(num_bus,num_bus);
\f2\fs26\fsmilli13333 \cf2 \
\pard\pardeftab720\sl320\partightenfactor0
\cf2 for k = 1:num_line
\f1\fs24 \

\f2\fs26\fsmilli13333 shuntadmittance = (Line(k,5)+i*Line(k,6)); 
\f1\fs24 \

\f2\fs26\fsmilli13333 Y_shunt(Line(k,1),Line(k,2)) = shuntadmittance;
\f1\fs24 \

\f2\fs26\fsmilli13333 Y_shunt(Line(k,2),Line(k,1)) = shuntadmittance;
\f1\fs24 \

\f2\fs26\fsmilli13333 end\
\pard\pardeftab720\sl280\partightenfactor0

\f0\fs24 \cf2 Y_mach = zeros(num_bus,1);\
\pard\pardeftab720\sl320\partightenfactor0

\f2\fs26\fsmilli13333 \cf2 for k = 1:num_mc
\f1\fs24 \

\f2\fs26\fsmilli13333 Y_mach(Machine(k,1)) = 1./( i*Machine(k,4));
\f1\fs24 \

\f2\fs26\fsmilli13333 end
\f1\fs24 \
\

\f2\fs26\fsmilli13333 Y_tl = zeros(num_bus,num_bus);
\f1\fs24 \

\f2\fs26\fsmilli13333 Y_mc = zeros(num_bus,num_bus);
\f1\fs24 \
\

\f2\fs26\fsmilli13333 for k = 1:num_bus
\f1\fs24 \

\f2\fs26\fsmilli13333 for j = k:num_bus
\f1\fs24 \

\f2\fs26\fsmilli13333 %Fill off-diagonal elements
\f1\fs24 \

\f2\fs26\fsmilli13333 if (k ~= j)
\f1\fs24 \

\f2\fs26\fsmilli13333 Y_tl(k,j) = -1*Y_line(k,j);
\f1\fs24 \

\f2\fs26\fsmilli13333 Y_tl(j,k) = Y_tl(k,j);
\f1\fs24 \

\f2\fs26\fsmilli13333 end
\f1\fs24 \
\

\f2\fs26\fsmilli13333 %Fill diagonal elements
\f1\fs24 \

\f2\fs26\fsmilli13333 if (k == j)
\f1\fs24 \

\f2\fs26\fsmilli13333 % Find the sum of all admittances connected to Bus i 
\f1\fs24 \

\f2\fs26\fsmilli13333 Y_tl(k,j) = sum(Y_line(k,:))+(sum(Y_shunt(k,:))/2);
\f1\fs24 \

\f2\fs26\fsmilli13333 end 
\f1\fs24 \

\f2\fs26\fsmilli13333 end
\f1\fs24 \

\f2\fs26\fsmilli13333 end
\f1\fs24 \
\

\f2\fs26\fsmilli13333 % Fill Ybus\
for k = 1:num_bus\
for j = k:num_bus\
%Fill off-diagonal elements\
if (k ~= j)\
Y_mc(k,j) = -1*Y_line(k,j);\
Y_mc(j,k) = Y_mc(k,j);\
end\
%Fill diagonal elements\
if (k == j)\
% Find the sum of all admittances connected to Bus i \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\
Y_mc(k,j) = sum(Y_line(k,:))+(sum(Y_shunt(k,:))/2)+Y_mach(k,:);\
end \
end\
end
\f1\fs24 \
\
\pard\pardeftab720\sl320\partightenfactor0

\f2\fs26 \cf2 disp('Y bus - transmission lines only');\
disp(Y_tl);\
disp('Y bus - equipment impedance included');\
disp(Y_mc)\
\
\
}