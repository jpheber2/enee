function out = moment_u (image,r,s)
[row col]=size(image);

out=0;
for i=1:row
    for j=1:col
       out=out+((i-1)^r) * ((j-1)^s) * (image(i,j)); 
    end
end
end