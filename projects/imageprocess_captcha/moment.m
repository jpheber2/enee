function m = moment(image,p,q)

mask = ones(size(image,1),size(image,2));   %if mask is not specified, select the whole image


image = double(image);
m=0; 
for i=1:1:size(mask,1)
    for j=1:1:size(mask,2)
        if mask(i,j) == 1
            m = m + (double((image(i,j))*(i^p)*(j^q))) ; %moment calculation
        end
    end
end
