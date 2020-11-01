function ncc2=ncc2(f,g)

%if ndims(f)==3
%    f=rgb2gray(f);
%end

%if ndims(g)==3
%    g=rgb2gray(g);
%end

[M,N]=size(f);
[H,L]=size(g);

ff=uint8(zeros(M+2*H-2,N+2*L-2));
ff(H:H+M-1,L:L+N-1)=f;

for ii=1:M+H-1
    for jj=1:N+L-1
        f1=ff(ii:ii+H-1,jj:jj+L-1);
        g=flipud(fliplr(g));
        ncc2(ii,jj)=ncc_calc(f1,g);
    end 
end

end

function out=ncc_calc(f1,g)

[Z,W]=size(f1);

norm_f1=f1-mean2(f1);
norm_g=g-mean2(g);

out=sum(norm_f1(:).*norm_g(:))/sqrt(sum(norm_f1(:).^2)*sum(norm_g(:).^2));

end

