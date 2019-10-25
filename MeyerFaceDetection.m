% David Meyer Face Detection 10/22/2019
clc
clear all
close all
%% Enhancement
im_pro =imread('person01_01d.png');
im_en=histeq(im_pro); %GrayScale, no black. Brighten it up!
%figure,montage({im_pro,im_en}) %Visual representation for me to see what's
                                %going on.
%% Training
%Lets get the training set
trainList=dir('./enroll/*.png');
im = imread(['./enroll/',trainList(1).name]); %Put info into Categories, that's awesome. So it's an object now?
%Information we know about the data set, num of images and num of people
%used to train
[r,c]=size(im);
numOfImages=length(trainList);
numOfPeople=numOfImages/2;
%eigen vector setup...
x=zeros(r*c,numOfPeople);%Is this A? I think so, 50^2 = 2500. Just need P2 now
vectorOfPeps=zeros(r*c,numOfImages);%This is the eigenvector setup.

Mec=zeros(r*c,1);%Average face, this is frakeinstein, fear the amazing personality.
index=zeros;index2=zeros;
match=zeros(1,10);%What is this for?
match2=zeros(1,10);
cmc=zeros(1,10);
cmc2=zeros(1,10);

%% Convert to vectors
%%%%%% convert all images to vector %%%%%%
for i=1:numOfImages
im =imread(['./enroll/',trainList(i).name]);
vectorOfPeps(:,i)=reshape(im',r*c,1); % Has all the image info
end
%% Get Xi and Me
j=1;
for i=1:2:(numOfImages-1)%Change the number 2 here to how many people/pictures
    x(:,j)=(vectorOfPeps(:,i)+vectorOfPeps(:,i+1))./2;%Picture 1 and 2 = person1, pic 3 and 4 = person2
    Mec(:,1)=Mec(:,1)+vectorOfPeps(:,i)+vectorOfPeps(:,i+1);
    j=j+1;
end
Me = Mec(:,1) ./ numOfImages;% The four different people

%% Get big A

for i=1:numOfPeople
    a(:,i)=x(:,i) - Me;  %Average of person i
end

%% Change to A to P2 for easier computations for the computer
ata = a'*a;  
[V D] = eig(ata);%eig = eigenvectors
p2 = [];
for i = 1 : size(V,2) 
    if( D(i,i)>1 )
        p2 = [p2 V(:,i)];
    end
end



%% TURN UP THE WEIGHTS!!!
wta=p2'*ata; % A*P2= P;  P'*A =Wt_A
plot(wta(:,1));  title('Weights representing Faces of Person1');
figure,plot(wta(:,2));  title('Weights representing Faces of Person2');
figure,plot(wta(:,3));  title('Weights representing Faces of Person3');
figure,plot(wta(:,4));  title('Weights representing Faces of Person4');
            %Need to add the rest of the people to show the weights, but
            %what's really the point? This is just for visualization

            
%% Get the Eigenfaces    
ef =a*p2;  %here is the P you need to use in matching 
[rr,cc]=size(ef);

for i=1:cc
eigim_t=ef(:,i);
eigface(:,:,i)=reshape(eigim_t,r,c);

figure,imagesc(eigface(:,:,i)');

axis image;axis off; colormap(gray(256));
title('Eigen Face Image','fontsize',10);
end
            
            
            
