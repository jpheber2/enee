function ncc1=ncc1(f,g)

%f=rgb2gray(f);
%g=rgb2gray(g);

[M,N]=size(f);
[H,L]=size(g);

ff=uint8(zeros(M+H-1,N+L-1));
ff(1:M,1:N)=f;

for ii=1:M
    for jj=1:N
        f1=ff(ii:ii+H-1,jj:jj+L-1);
        ncc1(ii,jj)=ncc_calc(f1,g);
    end
end

end

function out=ncc_calc(f1,g)

[Z,W]=size(f1);

norm_f1=f1-mean2(f1);
norm_g=g-mean2(g);

out=sum(norm_f1(:).*norm_g(:))/sqrt(sum(norm_f1(:).^2)*sum(norm_g(:).^2));

end

