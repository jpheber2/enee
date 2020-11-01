close all;
clear all;
clc;

img=imread('cameraman.tif');
img=double(img);
[M N]=size(img);

V=zeros((M*N),64);

x=1;
y=1;

theta_max=pi;        % orientation
f=sqrt(2);           % frequency
sigma=pi/3;          % sigma

for j = 1:4                          % varying value of std
    for v = 1:4                      % varying frequency
        for u = 1:4                  % varying orientation angle
          
        GW = zeros ( M , N );                          
        theta = ( theta_max / ( f ^ v ) ) * exp( 1i * u * pi / 8 );
        theta2 = ( abs( theta ) ) ^ 2;

        for m = -M/2+1 : M/2
            for n = -N/2+1 : N/2

            GW(m+M/2,n+N/2) = ( theta2 / sigma ) * exp( -0.5 * theta2 * ( m ^ 2 + n ^ 2 ) / sigma) * ( exp( 1i * ( real( theta ) * m + imag ( theta ) * n ) ) - exp ( -0.5 * sigma ) );

            end
        end
                                    
        figure( j );
        subplot( 4, 4, y ),imshow ( real( GW ),[] ); % Show the Gabor wavelets
        xlim( [96,160] ); ylim( [96,160] );
 
        P=conv2(img,double(GW),'same');

        V(:,x)=P(:);
        x=x+1;
        y=y+1;
        
        end
    end
sigma=(j+1)*sigma;
y=1;
end


%%

clc
clust = zeros(size(V,1),6);
for i=1:6                                               % evaluation for k
clust(:,i) = kmeans(real(V),i,'emptyaction','singleton','replicate',5);
end
va = evalclusters(real(V),clust,'CalinskiHarabasz');    % selection of k

k=va.OptimalK        % k selected

indexes = kmeans(real(V), k);
rows=256;
columns=256;

if k==2
class1 = reshape(indexes == 1, rows, columns);
class2 = reshape(indexes == 2, rows, columns);
allClasses = cat(3, class1, class2);

    else if k==3
    class1 = reshape(indexes == 1, rows, columns);
    class2 = reshape(indexes == 2, rows, columns);
    class3 = reshape(indexes == 3, rows, columns);
    allClasses = cat(3, class1, class2, class3);

        else if k==4
        class1 = reshape(indexes == 1, rows, columns);
        class2 = reshape(indexes == 2, rows, columns);
        class3 = reshape(indexes == 3, rows, columns);
        class4 = reshape(indexes == 4, rows, columns);
        allClasses = cat(3, class1, class2, class3, class4);
        
            else if k==5
            class1 = reshape(indexes == 1, rows, columns);
            class2 = reshape(indexes == 2, rows, columns);
            class3 = reshape(indexes == 3, rows, columns);
            class4 = reshape(indexes == 4, rows, columns);
            class5 = reshape(indexes == 5, rows, columns);
            allClasses = cat(3, class1, class2, class3, class4, class5);
            
            else 
            class1 = reshape(indexes == 1, rows, columns);
            class2 = reshape(indexes == 2, rows, columns);
            class3 = reshape(indexes == 3, rows, columns);
            class4 = reshape(indexes == 4, rows, columns);
            class5 = reshape(indexes == 5, rows, columns);
            class6 = reshape(indexes == 6, rows, columns);
            allClasses = cat(3, class1, class2, class3, class4, class5, class6);
                
           end
       end
   end
end

allClasses = allClasses(:, :, 1:k); 

plotRows = ceil(sqrt(size(allClasses, 3)));

indexedImage = zeros(rows, columns, 'uint8'); 

figure();

for c = 1 : k
	
	subplot(plotRows, plotRows, c);
	thisClass = allClasses(:, :, c);
	imshow(thisClass);
	
end
