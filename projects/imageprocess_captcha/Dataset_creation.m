clc
clear all
close all

a=imread('AtoZ.bmp');
img=rgb2gray(a);
r=1;
for i=1:26;
c=r+29;
x=img(:,r:c);
r=c+1;
imwrite(x,strcat('Alphabet',num2str(i),'.png'));
end

%%
clc
clear all
close all

a=imread('1to0.bmp');
img=rgb2gray(a);
r=1;
for i=1:10;
c=r+29;
x=img(:,r:c);
r=c+1;
imwrite(x,strcat('Num',num2str(i),'.png'));
end
