function eta = SI_Moment(image)


image = double(image);

    mask = ones(size(image,1),size(image,2));


% computation of central moments upto order order 3
for i=1:1:4
    for j=1:1:4
        nu(i,j) = Centr_Moment(image, mask,i-1,j-1);
    end
end

% computation of scale invariant moments using central monets of upto order

eta = zeros(3,3);
for i=1:1:4
    for j=1:1:4
        if i+j >= 4
            eta(i,j) = (double(nu(i,j))/(double(nu(1,1)).^(double((i+j)/2)))); %scale invariant moment matrix
        end
    end
end