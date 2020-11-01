% main point in this is that value of p q and l should be similar.
clc
clear all
close all
N=26;

Image='Alphabet%d.png';
p=randi(26,1,5);
for i=1:5
image=imread(sprintf(Image,p(i)));
figure();
imshow(image)
r=rand;
Resize=imresize(image,1);
Res=Affine_fun(Resize)
load('Afine_alphabet')
load('Alphabet_Num')
for j=1:26
    am= (Res-Affine_alphabet(:,j))
    am(isnan(am)) = 0;
an(j,:)=sum(am');
%[D I] = pdist2(Res,Affine_alphabet(:,j),'euclidean')
end
[M,I] = min(abs(an));
q(i)=I;
Rotated=imrotate(image,1);
Res=Affine_fun(Rotated)
load('Afine_alphabet')
load('Alphabet_Num')
for j=1:26
    am= (Res-Affine_alphabet(:,j))
    am(isnan(am)) = 0;
an(j,:)=sum(am');
%[D I] = pdist2(Res,Affine_alphabet(:,j),'euclidean')
end
[M,I] = min(abs(an));
q(i)=I;

shift_x=imtranslate(image,[12, 0]);
shift_y=imtranslate(image,[0, 8.5]);

J = imnoise(image,'gaussian',0,0.0001)

Res=Affine_fun(J)
load('Afine_alphabet')
load('Alphabet_Num')
for j=1:26
    am= (Res-Affine_alphabet(:,j))
    am(isnan(am)) = 0;
an(j,:)=sum(am');
%[D I] = pdist2(Res,Affine_alphabet(:,j),'euclidean')
end
[M,I] = min(abs(an));
q(i)=I;


J =  imnoise(image,'salt & pepper', 0.0001);

Res=Affine_fun(J)
load('Afine_alphabet')
load('Alphabet_Num')
for j=1:26
    am= (Res-Affine_alphabet(:,j))
    am(isnan(am)) = 0;
an(j,:)=sum(am');
%[D I] = pdist2(Res,Affine_alphabet(:,j),'euclidean')
end
[M,I] = min(abs(an));
l(i)=I;


end

close all
%%
