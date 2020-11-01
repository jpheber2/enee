function cm = central_moment(image,x_hat,y_hat,r,s)

[r c]=size(image);

cm=0;
for i=1:r
    for j=1:c
       cm=cm+((i-1-x_hat)^r) * ((j-1-y_hat)^s) * (image(i,j)); 
    end
end
