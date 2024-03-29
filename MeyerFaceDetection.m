% David Meyer    Face Detection     10/22/2019
clc
clear all
close all

chloe1a = imread('Dataset/enrolling/ID45_001.png');
chloe2a = imread('Dataset/enrolling/ID45_002.png');
chloe3a = imread('Dataset/enrolling/ID45_003.png');
chloe4a = imread('Dataset/enrolling/ID45_004.png');
chloe5a = imread('Dataset/enrolling/ID45_005.png');


chloe1 = rgb2gray(chloe1a);%100X68 need to change size
chloe1 = imresize(chloe1, [100 100]);
chloe2 = rgb2gray(chloe2a);
chloe2 = imresize(chloe2, [100 100]);
chloe3 = rgb2gray(chloe3a);
chloe3 = imresize(chloe3, [100 100]);
chloe4 = rgb2gray(chloe4a);
chloe5 = rgb2gray(chloe5a);

%figure,montage({chloe5a,chloe5})
%figure,imshow(chloe1)
%figure,imshow(chloe2)
%figure,imshow(chloe3)
%figure,imshow(chloe4)
%figure,imshow(chloe5)

imwrite(chloe1, 'ID45_001.bmp')
imwrite(chloe2, 'ID45_002.bmp')
imwrite(chloe3, 'ID45_003.bmp')
imwrite(chloe4, 'ID45_004.bmp')
imwrite(chloe5, 'ID45_005.bmp')
%Now my pictures are similar to all the other pictures inside the dataset.
%Manually moved them to the folder to use in the enrolling and training.



%NOTES OF GIVEN SET
%All images 100 X 100 and grayscaled, so already normalized.
%Only have to find the characteristics using eigenvectors. 
%IFF AA' = 0, then same characteristics meaning same person.

%No ID28_###.bmp .   SO... Only 43 people

%% Enhancement
im_pro =imread('Dataset/testing/ID01_010.bmp');%Could encapsulate imread section, it works
im_en=histeq(im_pro); %GrayScale, no black. Brighten it up!
%figure,montage({im_pro,im_en}) %Visual representation for me to see what's
                                %going on.
                                
                                %Was she just showing that this is where we
                                %would have to enhance using the histogram
                                %equalization?
                                
                                
%% Training
%Lets get the training set
trainList=dir('Dataset/enrolling/*.bmp');%Loop through folder making list
im = imread(['Dataset/enrolling/',trainList(1).name]);
imHist = histeq(im);
[r,c]=size(imHist);

numOfImages=length(trainList);
numOfPeople=numOfImages/5;%Divide by 5 because they're are 5 pictures of each person

%eigen vector setup...
x=zeros(r*c,numOfPeople);%This is incorrect
vectorOfPeps=zeros(r*c,numOfImages);%This is the eigenvector setup.

Mec=zeros(r*c,1);%Average face, this is frakeinstein, fear the amazing personality.
index=zeros;
index2=zeros;

match=zeros(1,10);%What is this for?
match2=zeros(1,10);
cmc=zeros(1,10);
cmc2=zeros(1,10);



%% Convert to vectors
%%%%%% convert all images to vector %%%%%%
for i=1:numOfImages
    im =(imread(['Dataset/enrolling/',trainList(i).name]));%Histogram equalization here to be more accurate once it works
    vectorOfPeps(:,i)=reshape(im',r*c,1); 
end
%% Get Xi and Me
j=1;
for i=1:5:(numOfImages-4)%Change the number 2 here to how many people/pictures
    %minus 1 why?
    x(:,j)=( vectorOfPeps(:,i)+vectorOfPeps(:,i+1)+vectorOfPeps(:,i+2)+...
        vectorOfPeps(:,i+3)+vectorOfPeps(:,i+4) ) ./ 5;%Mean Picture
    
    Mec(:,1)=Mec(:,1)+vectorOfPeps(:,i)+vectorOfPeps(:,i+1)+...
        vectorOfPeps(:,i+2)+vectorOfPeps(:,i+3)+vectorOfPeps(:,i+4);%Divide here, coincidence?
    j=j+1;
end
Me = Mec(:,1) ./ numOfImages; %The Average vector of all the enrolled images

%% Get big A

for i=1:numOfPeople
    a(:,i)=x(:,i) - Me;  %A, the matrix with the all the face vectors after subtracting the mean vector
end

%% Change to A to P2 for easier computations for the computer
%c = AA'    '(this is transpose)   Dimentionality Reduction
%ata represents ...
ata = a' * a;  
[V D] = eig(ata);%eig = eigenvectors
    %V eigenvector matrix
    %D is the eigenvalue matrix (diagonal) RowEchilonForm?
    
p2 = [];
for i = 1 : size(V,2) 
    if( D(i,i)>1 )
        
        p2 = [p2 V(:,i)];
    end
end



%% TURN UP THE WEIGHTS!!!
wta = p2'* ata; % A*P2= P;  P'*A =Wt_A      


            
%% Get the Eigenfaces    
ef = a * p2;  %Here is the P you need to use in matching 


%In this part of recognition, we compare two faces by projecting the images into facespace and 
% measuring the Euclidean distance between them.
%
%               index      -   the recognized image name
%             imlist2      -   the path of test image
%                Me        -   mean image vector
%                a         -   mean subtracted image vector matrix
%                ef         -   eigenfaces that are calculated from eigenface function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



[rr,cc]=size(ef);

for i=1:cc
    eigim_t = ef(:,i);%Projected image
    eigface(:,:,i)=(reshape(eigim_t,r,c));%Extract PCA features

    figure,imagesc(eigface(:,:,i)');
    axis image;axis off; colormap(gray(256));
    title('Eigen Face Image','fontsize',10);
end
    


%%
%%%%%%%%%%%%%%%%%%%%%%%  TESTING  %%%%%%%%%%%%%%%%%%%%%%%%
imlist2=dir('Dataset/testing/*.bmp');
numOfImages=length(imlist2);

imgTesting_vector=zeros(r*c,numOfImages);


%%
%%%%%% Convert all test images to vector %%%%%%
for i=1:numOfImages
    im =(imread(['Dataset/testing/',imlist2(i).name]));%Histeq here?
    imgTesting_vector(:,i)=(reshape(im',r*c,1)); %Extract PCA features
    
    %%%%% get B=y-me %%%%%%%
    b(:,i)=imgTesting_vector(:,i) - Me;  %% bi=imt_vector(i)-Me;
    wtb = ef' * b(:,i);  %%wtb=P'*bi;  %What does this stand for?
    %How did I end up with 20 people?
    %figure,plot(wtb); title('Weights representing Faces of Person:wtb');
    
    for ii=1:numOfPeople
        %Calculate the euclidian distance of all projectedtrained image
        %from projected test image
        
        eud(ii)=sqrt(sum(norm(wtb-wta(:,ii)).^2));%euclidian distance
        
        %wtb is already set to person i, but we have to get wta(:,i)
        %figure,plot(eud)
        
    end
    
    [cdata index(i)] = min(eud);%% smallest Euclide distance this is match         
                                %cdata= distance, index = recognized
    
    
    %%%%%%%%%%%%%%%%%%%%%%%  RESULT  %%%%%%%%%%%%%%%%%%%%%%%%
    rresult=[1 2 3 4 5 6 7 8 9 10 ];
   
    %%%%%%%%%%%%%%% CMC calculation %%%%%%%
    
    
    %Only compares to imlist2
    if index(i)==rresult(i) %The pictures are to different, and it's easy to confuse the data with more people. 
        match(1)=match(1)+1;%%%%%%%first rank matching number
        
        count = i;
        count
        match
        i
    else
        [svals,idx]=sort(eud(:));
        index2(i)=idx(2);
        if index2(i)==rresult(i)
            match(2)=match(2)+1;%%%%%%%second rank matching number       
        end 
    end
end  
%%
for i=1:10   
    cmc(i)=sum(match(1:i)) / numOfImages;
end



%% VISUALS
%Weights plots for each person
figure,plot(wta(:,1));  title('Weights representing Faces of Person1');
figure,plot(wta(:,2));  title('Weights representing Faces of Person2');
figure,plot(wta(:,3));  title('Weights representing Faces of Person3');
figure,plot(wta(:,4));  title('Weights representing Faces of Person4');
figure,plot(wta(:,5));  title('Weights representing Faces of Person5');
figure,plot(wta(:,6));  title('Weights representing Faces of Person6');
figure,plot(wta(:,7));  title('Weights representing Faces of Person7');
figure,plot(wta(:,8));  title('Weights representing Faces of Person8');
figure,plot(wta(:,9));  title('Weights representing Faces of Person9');
figure,plot(wta(:,10)); title('Weights representing Faces of Person10');
figure,plot(wta(:,11)); title('Weights representing Faces of Person11');
figure,plot(wta(:,12)); title('Weights representing Faces of Person12');
figure,plot(wta(:,13)); title('Weights representing Faces of Person13');
figure,plot(wta(:,14)); title('Weights representing Faces of Person14');
figure,plot(wta(:,15)); title('Weights representing Faces of Person15');
figure,plot(wta(:,16)); title('Weights representing Faces of Person16');
figure,plot(wta(:,17)); title('Weights representing Faces of Person17');
figure,plot(wta(:,18)); title('Weights representing Faces of Person18');
figure,plot(wta(:,19)); title('Weights representing Faces of Person19');
figure,plot(wta(:,20)); title('Weights representing Faces of Person20');
figure,plot(wta(:,21)); title('Weights representing Faces of Person21: Chloe');

            
%CMC CURVE            
figure,plot(cmc)
title('CMC curve')




%% Self pondering(Just happened to be where I took my notes for this phone call)
%More math
%linear algebra
%recommend py, adv math, 6-7 hours, $3600
%cornel professor Wienburger
%Do I really want to take this course? Stuff is cool, just don't know if I
    %want a career with it?

    