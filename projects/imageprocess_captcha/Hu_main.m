clc
clear all
close all
format long
N=26;

Image='Alphabet%d.png';

hu_mv_string=zeros([7 1 N]);
hu_main=zeros(7,N);

for i=1:N 
    image=imread(sprintf(Image,i));
    
    imgbw=im2bw(image);
    imgbw=logical(imgbw);
    figure,imshow(imgbw,[]);

    eta = SI_Moment(image)

inv_moments = Hu_Moments(eta)

    hu_mv_string(:,:,i)=inv_moments;
    hu_main(:,i)=hu_mv_string(:,:,i);   
end
Alphabet_main=hu_main;
save Alphabet_Num Alphabet_main

