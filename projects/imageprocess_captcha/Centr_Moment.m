function cen_mmt = Centr_Moment(image,mask,p,q)

    mask = ones(size(image,1),size(image,2)); %if mask is not spcified, select the whole image


image = double(image);

%moments necessary to compute components of centroid
m10 = moment(image,1,0); 
m01 = moment(image,0,1);
m00 = moment(image,0,0);

%components of centroid
x_cen = floor(m10/m00);
y_cen = floor(m01/m00);

cen_mmt =0;

for i=1:r
    for j=1:c
        cen_mmt = cen_mmt + ((i-x_cen)^p)*((j-y_cen)^q)*image(i,j)); %calculating central moment
    end
end
