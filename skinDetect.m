%% 
% Given Code From blackboard
function skin = skinDetect(pic)
    %given code in the assignment details.
    %All this does is detect skin pixels using RGB
    ims1 = (pic(:,:,1)>95) & (pic(:,:,2)>40) & (pic(:,:,3)>20);
    ims2 = (pic(:,:,1)-pic(:,:,2)>15) | (pic(:,:,1)-pic(:,:,3)>15);
    ims3 = (pic(:,:,1)-pic(:,:,2)>15) & (pic(:,:,1)>pic(:,:,3));
    ims = ims1 & ims2 & ims3;
    %figure,imshow(ims)
    skin = ims;
end