%% Detect Chloe's face
clc 
clear all
close all

picPath = 'Dataset/enrolling/ID45_001.png';

chloePic = imread(picPath);

skinGood = skinDetect(picPath);
figure,imshow(skinGood)
skinGood = uint8(skinGood);