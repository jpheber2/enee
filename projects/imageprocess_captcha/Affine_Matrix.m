clc
clear all
close all
N=26;

Image='Alphabet%d.png';

Affine_string=zeros([7 1 N]);
Affine_main=zeros(7,N);
for i=1:N 
    image=imread(sprintf(Image,i));
    
    imgbw=im2bw(image);
    imgbw=logical(imgbw);
    figure,imshow(imgbw,[]);
    image=imgbw;
    gm10=moment_u(image,1,0);
gm01=moment_u(image,0,1);
m00=sum(sum(image));

x_hat=(gm10)/m00;
y_hat=(gm01)/m00;

mu_00 = sum(sum(image));
mu_11 = central_moment( image ,x_hat,y_hat,1,1);
mu_13 = central_moment( image ,x_hat,y_hat,1,3);
mu_20 = central_moment( image ,x_hat,y_hat,2,0);
mu_02 = central_moment( image ,x_hat,y_hat,0,2);
mu_21 = central_moment( image ,x_hat,y_hat,2,1);
mu_22 = central_moment( image ,x_hat,y_hat,2,2);
mu_12 = central_moment( image ,x_hat,y_hat,1,2);
mu_03 = central_moment( image ,x_hat,y_hat,0,3);
mu_30 = central_moment( image ,x_hat,y_hat,3,0);
mu_31 = central_moment( image ,x_hat,y_hat,3,1);
mu_40 = central_moment( image ,x_hat,y_hat,4,0);
mu_04 = central_moment( image ,x_hat,y_hat,0,4);


I1=(mu_20 * mu_02 - mu_11^2)/(mu_00)^4;
I2=(-(mu_30^2) * (mu_03^2) + 6*mu_30*mu_21*mu_12*mu_03 - 4*mu_30* (mu_12)^3)/(mu_00)^10;
I3=(mu_20 * mu_21 * mu_03 - mu_20 * (mu_12)^2 - mu_11 * mu_30  * mu_03 + mu_11 * mu_21 * mu_12 + mu_02 * mu_30 * mu_12 - mu_02 * (mu_21)^2)/(mu_00)^7;
I4=(-(mu_20)^3 * (mu_03)^2 + 6 * (mu_20)^2 * mu_11 * mu_12 * mu_03 - 3 * (mu_20)^2 * mu_02 * (mu_12)^2 -6 * mu_20 * (mu_11)^2 * mu_21 * mu_03 - 6 * mu_20 * mu_11^2 * mu_12^2 + 12 * mu_20 * mu_11 * mu_02 * mu_21 * mu_12 - 3 * mu_20 * mu_02^2 * mu_21^2 + 2 * mu_11^3 * mu_30 * mu_03 + 6 * mu_11^3 * mu_21 * mu_12 -6 * mu_11^2 * mu_02 * mu_30 * mu_12 -6 * mu_11^2 * mu_02 * mu_21^2 + 6 * mu_11 * mu_02^2 * mu_30 * mu_21 - 1 * mu_02^3 * mu_30^2)/(mu_00)^11;
I5=(1 * mu_40 * mu_04 -4 * mu_31 * mu_13 + 3 * mu_22^2)/(mu_00)^6;
I6=(1 * mu_40 * mu_22 * mu_04 -1 * mu_40 * mu_13^2 -1 * mu_31^2 * mu_04 + 2 * mu_31 * mu_22 * mu_13- 1* mu_22^3)/( mu_00)^9;
I7=(1 * mu_20^2 * mu_04 -4* mu_20 * mu_11 * mu_13 +2 * mu_20 * mu_02 * mu_22 + 4 * mu_11^2 * mu_22 - 4 * mu_11 * mu_02 * mu_31 + 1 * mu_02^2 * mu_40)/( mu_00)^7;

affine_moment_vector = [I1 I2 I3 I4 I5 I6 I7];

    Affine_string(:,:,i)=affine_moment_vector;
    Affine_main(:,i)=Affine_string(:,:,i);   
end
Affine_Alphabet=Affine_main;
save Afine_Alphabet Affine_Alphabet