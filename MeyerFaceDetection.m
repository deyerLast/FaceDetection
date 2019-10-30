% David Meyer Face Detection 10/22/2019
clc
clear all
close all

%chloe1 = imread('Dataset/enrolling/ID45_001.png');
%chloe2 = imread('Dataset/enrolling/ID45_002.png');
%chloe3 = imread('Dataset/enrolling/ID45_003.png');
%chloe4 = imread('Dataset/enrolling/ID45_004.png');
%chloe5 = imread('Dataset/enrolling/ID45_005.png');

%chloe1 = rgb2gray(chloe1);
%chloe1 = imresize(chloe1, [100 100]);
%chloe2 = rgb2gray(chloe2);
%chloe2 = imresize(chloe2, [100 100]);
%chloe3 = rgb2gray(chloe3);
%hloe3 = imresize(chloe3, [100 100]);
%chloe4 = rgb2gray(chloe4);
%chloe5 = rgb2gray(chloe5);
%figure,imshow(chloe1)
%figure,imshow(chloe2)
%figure,imshow(chloe3)
%figure,imshow(chloe4)
%figure,imshow(chloe5)

%chloe = imread('Dataset/enrolling/ID45_001.bmp');
%figure,imshow(chloe)
%NOTES
%All images 100X100, so already normalized.
%Only have to find the characteristics using eigenvectors. 
%IFF AA' = 0, then same characteristics meaning same person.

%No ID28_###.bmp .   SO... Only 43 people

%% Enhancement
im_pro =imread('Dataset/testing/ID01_010.bmp');%Could encapsulate imread section, it works
im_en=histeq(im_pro); %GrayScale, no black. Brighten it up!
%figure,montage({im_pro,im_en}) %Visual representation for me to see what's
                                %going on.
%% Training
%Lets get the training set
trainList=dir('Dataset/enrolling/*.bmp');%This is really cool that you can make a list this way
im = imread(['Dataset/enrolling/',trainList(1).name]);
[r,c]=size(im);
numOfImages=length(trainList);
numOfPeople=numOfImages/5;%Divide by 5 because they're are 5 pictures of each person

%eigen vector setup...
x=zeros(r*c,numOfPeople);%This is incorrect
vectorOfPeps=zeros(r*c,numOfImages);%This is the eigenvector setup.

Mec=zeros(r*c,1);%Average face, this is frakeinstein, fear the amazing personality.
index=zeros;
index2=zeros;

match=zeros(1,43);%What is this for?
match2=zeros(1,43);
cmc=zeros(1,10);
cmc2=zeros(1,10);

%% Convert to vectors
%%%%%% convert all images to vector %%%%%%
for i=1:numOfImages
    im =imread(['Dataset/enrolling/',trainList(i).name]);
    vectorOfPeps(:,i)=reshape(im',r*c,1); % Has all the image info
end
%% Get Xi and Me
j=1;
for i=1:5:(numOfImages-1)%Change the number 2 here to how many people/pictures
    %minus 1 why?
    x(:,j)=( vectorOfPeps(:,i)+vectorOfPeps(:,i+1)+vectorOfPeps(:,i+2)+...
        vectorOfPeps(:,i+3)+vectorOfPeps(:,i+4) )./5;%Mean Picture
    
    Mec(:,1)=Mec(:,1)+vectorOfPeps(:,i)+vectorOfPeps(:,i+1)+...
        vectorOfPeps(:,i+2)+vectorOfPeps(:,i+3)+vectorOfPeps(:,i+4);%Mean Vector
    j=j+1;
end

Me = Mec(:,1) ./ numOfImages;% The different people

%% Get big A

for i=1:numOfPeople
    a(:,i)=x(:,i) - Me;  %Average of person i
end

%% Change to A to P2 for easier computations for the computer
ata = a'*a;  
[V D] = eig(ata);%eig = eigenvectors   The diagonal of the matrix RowEchilonForm
    %V should have the same first column, but it doesn't. Why?

    
    
    
    
    
    
    %I THINK THIS MIGHT BE WRONG FOR THIS NOW
p2 = [];
for i = 1 : size(V,2) 
    if( D(i,i)>1 )
        p2 = [p2 V(:,i)];
    end
end



%% TURN UP THE WEIGHTS!!!
wta=p2'*ata; % A*P2= P;  P'*A =Wt_A

%Everything below this just shows a visual representation of the strong
%markers of each pictures.

%figure,plot(wta(:,1));  title('Weights representing Faces of Person1');
%figure,plot(wta(:,2));  title('Weights representing Faces of Person2');
%figure,plot(wta(:,3));  title('Weights representing Faces of Person3');
%figure,plot(wta(:,4));  title('Weights representing Faces of Person4');
%figure,plot(wta(:,5));  title('Weights representing Faces of Person5');
%figure,plot(wta(:,6));  title('Weights representing Faces of Person6');
%figure,plot(wta(:,7));  title('Weights representing Faces of Person7');
%figure,plot(wta(:,8));  title('Weights representing Faces of Person8');
%figure,plot(wta(:,9));  title('Weights representing Faces of Person9');
%figure,plot(wta(:,10)); title('Weights representing Faces of Person10');
%figure,plot(wta(:,11)); title('Weights representing Faces of Person11');
%figure,plot(wta(:,12)); title('Weights representing Faces of Person12');
%figure,plot(wta(:,13)); title('Weights representing Faces of Person13');
%figure,plot(wta(:,14)); title('Weights representing Faces of Person14');
%figure,plot(wta(:,15)); title('Weights representing Faces of Person15');
%figure,plot(wta(:,16)); title('Weights representing Faces of Person16');
%figure,plot(wta(:,17)); title('Weights representing Faces of Person17');
%figure,plot(wta(:,18)); title('Weights representing Faces of Person18');
%figure,plot(wta(:,19)); title('Weights representing Faces of Person19');
%figure,plot(wta(:,20)); title('Weights representing Faces of Person20');
%figure,plot(wta(:,21)); title('Weights representing Faces of Person21');
%figure,plot(wta(:,23)); title('Weights representing Faces of Person23');
%figure,plot(wta(:,24)); title('Weights representing Faces of Person24');
%figure,plot(wta(:,25)); title('Weights representing Faces of Person25');
%figure,plot(wta(:,26)); title('Weights representing Faces of Person26');
%figure,plot(wta(:,27)); title('Weights representing Faces of Person27');
%figure,plot(wta(:,28)); title('Weights representing Faces of Person28');
%figure,plot(wta(:,29)); title('Weights representing Faces of Person29');
%figure,plot(wta(:,30)); title('Weights representing Faces of Person30');
%figure,plot(wta(:,31)); title('Weights representing Faces of Person31');
%figure,plot(wta(:,32)); title('Weights representing Faces of Person32');
%figure,plot(wta(:,33)); title('Weights representing Faces of Person33');
%figure,plot(wta(:,34)); title('Weights representing Faces of Person34');
%figure,plot(wta(:,35)); title('Weights representing Faces of Person35');
%figure,plot(wta(:,36)); title('Weights representing Faces of Person36');
%figure,plot(wta(:,37)); title('Weights representing Faces of Person37');
%figure,plot(wta(:,38)); title('Weights representing Faces of Person38');
%figure,plot(wta(:,39)); title('Weights representing Faces of Person39');
%figure,plot(wta(:,40)); title('Weights representing Faces of Person40');
%figure,plot(wta(:,41)); title('Weights representing Faces of Person41');
%figure,plot(wta(:,42)); title('Weights representing Faces of Person42');
%figure,plot(wta(:,43)); title('Weights representing Faces of Person43');
%figure,plot(wta(:,44)); title('Weights representing Faces of Person44');
%This for loop didn't plot for some reason.
%for number = 20:43
%    figure,plot(wta(:,number)); title('Weights representing Faces of Person' + number);
%end
            %Need to add the rest of the people to show the weights, but
            %what's really the point? This is just for visualization

            
%% Get the Eigenfaces    
ef =a*p2;  %here is the P you need to use in matching 
[rr,cc]=size(ef);

for i=1:cc
    eigim_t=ef(:,i);
    eigface(:,:,i)=reshape(eigim_t,r,c);

    %figure,imagesc(eigface(:,:,i)');

    axis image;axis off; colormap(gray(256));
    title('Eigen Face Image','fontsize',10);
end
     
%%
%%%%%%%%%%%%%%%%%%%%%%%  TESTING  %%%%%%%%%%%%%%%%%%%%%%%%
imlist2=dir('Dataset/testing/*.bmp');
numOfImages=length(imlist2);
imt_vector=zeros(r*c,numOfImages);


%%
%%%%%% convert all test images to vector %%%%%%
for i=1:numOfImages
    im =histeq(imread(['Dataset/testing/',imlist2(i).name]));
    imt_vector(:,i)=reshape(im',r*c,1);
    
    %%%%% get B=y-me %%%%%%%
    b(:,i)=imt_vector(:,i)-Me;  %% bi=imt_vector(i)-Me;
    wtb=ef'*b(:,i);  %%wtb=P'*bi;
    
    for ii=1:numOfPeople   %% weight compare wtb and wta(i)
        eud(ii)=sqrt(sum((wtb-wta(:,ii)).^5));%Changed from .^2
    end
    [cdata index(i)]=min(eud);  %% find minimum eud's index .         cdata Does What?

       %%%%%%%%%%%%%%%%%%%%%%%  RESULT  %%%%%%%%%%%%%%%%%%%%%%%%
    %%% right result by observation is 1 1 2 3 4 %%%%%
    rresult=[1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 5 5 5 5 5 6 6 6 6 6 7 7 7 7 7 8 8 8 8 8 9 9 9 9 9 10 10 10 10 10 11 11 11 11 11 12 12 12 12 12 13 13 13 13 13 14 14 14 14 14 15 15 15 15 15 16 16 16 16 16 17 17 17 17 17 18 18 18 18 18 19 19 19 19 19 20 20 20 20 20 21 21 21 21 21 22 22 22 22 22 23 23 23 23 23 24 24 24 24 24 25 25 25 25 25 26 26 26 26 26 27 27 27 27 27 29 29 29 29 29 30 30 30 30 30 31 31 31 31 31 32 32 32 32 32 33 33 33 33 33 34 34 34 34 34 35 35 35 35 35 36 36 36 36 36 37 37 37 37 37 38 38 38 38 38 39 39 39 39 39 40 40 40 40 40 41 41 41 41 41 42 42 42 42 42 43 43 43 43 43 44 44 44 44 44];
    %fprintf(rresult)
    %%%%%%%%%%%%%%% CMC calculation %%%%%%%
    if index(i)==rresult(i)
        match(1)=match(1)+1;%%%%%%%first rank matching number
    else
        [svals,idx]=sort(eud(:));
        index2(i)=idx(2);
        if index2(i)==rresult(i)
            match(2)=match(2)+1;%%%%%%%second rank matching number
        end 
    end
end  
%%
for i=1:10  %% if show CMC of the 1st to 10th rank matching number 
    cmc(i)=sum(match(1:i))/numOfImages;
end
figure,plot(cmc);
title('CMC curve');
            
