function [out] = HU_fun(image)
 imgbw=im2bw(image);
    imgbw=logical(imgbw);
    %figure,imshow(imgbw,[]);

    eta = SI_Moment(image)

inv_moments = Hu_Moments(eta)

   out=inv_moments;
  
end

