%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This p code is an implementation of the paper:              %
% RGBD-Fusion: Real-Time High Precision Depth Recovery        %
% Roy Or-El, Guy Rosman, Aaron Wetzler, Ron Kimmel,           %
% Alfred M. Bruckstein. Proceedings of IEEE CVPR 2015         %
%                                                             %
% This code should be used for research purposes only.        %
% Any commercial use of this code is strictly prohibited.     %
% Any distribution of this code requires the authors'         %
% explicit premision in writing.                              %
%                                                             %
% Please cite our paper if you use this code in your          %
% publication.                                                %
%                                                             %
% Requirements                                                %
%                                                             %
% 1. Depth map and color image should have the same size.     %
% 2. Depth map and color image should be aligned.             %
% 3. Depth values should be between 0 and 1000.               %
%    i.e. if the largest depth value is 4630 divide 'z_im'    %
%    by 10, apply the algorithm and multiply 'surface' by 10. %
% 4. Missing depth values should be saved as 0                %
% 5. The *_depth.mat file should include a single 'double'    %
%    type variable named z_im that contains the depth map.    %
% 6. Color image should be an RGB image of type 'uint8'       %
% 7. The *_calib.mat file should store the 3x3 intrinsic      %
%    matrix of the depth camera. Variable should be named     %
%    'K'.                                                     %
% 8. The *_mask.mat file should store a binary 'logical'      %
%    mask of the object for rendering. Variable should be     %
%    named 'mask'                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load data\D_SZ_NM_N_F.mat
load data\mannequin_calib.mat
load data\mask_SZ_NM_N_F.mat
aligned_color = imread('data\C_SZ_NM_N_F.png');
sz = size(z_im);
% A (0,0,0) RGB value is treated as missing color data.
% If your color image does not have any missing data use this 
% function to correct any such values to (1,1,1).
% If your color image has missing data, please comment the line below.
aligned_color = correct_all_black(aligned_color,true(sz)); 

% Bilateral filter parameters
bilateral_params.w     = 9;       % bilateral filter half-width
bilateral_params.sigma = [8 15];  % bilateral filter standard deviations

% Albedo and specularities optimization parameters
lighting_opt_params.tau_c = 0.05;
lighting_opt_params.sigma_c = sqrt(0.05);
lighting_opt_params.sigma_d = sqrt(50);
lighting_opt_params.lambda_rho = 0.1; %Kinect - 10 IVCAM - 0.1
lighting_opt_params.lambda_beta1 = 1; 
lighting_opt_params.lambda_beta2 = 1;

% Depth optimization parameters
depth_opt_params.lambda_z1 = 0.004;
depth_opt_params.lambda_z2 = 0.0075;