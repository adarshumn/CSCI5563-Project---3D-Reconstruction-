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

clearvars;
close all
clc

prepare_data

surface = rgbd_fusion(z_im,aligned_color,sz,bilateral_params,lighting_opt_params,depth_opt_params);

% Render result
[X,Y,Z] = back_project(surface,K);
X(~mask) = nan; Y(~mask) = nan; Z(~mask) = nan;
X(Z==0) = nan; Y(Z==0) = nan; Z(Z==0) = nan;
h=surf(X,Z,-Y,Z);
set(h,'LineStyle','none')
axis image
grid off
box on
view(18, 0);
set(h,'FaceLighting','phong','FaceColor','interp',...
      'AmbientStrength',0.5)
light('Position',[-0.6163 -0.6654 0.4210],'Style','infinite');