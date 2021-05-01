z_im = imread("data\D_SZ_NM_N_F.png");
z_im = z_im(:, :, 1);
z_im = squeeze(z_im);
z_im = cast(z_im,'double');
save("data\D_SZ_NM_N_F.mat", 'z_im')

mask = imread("data\mask_SZ_NM_N_F.png");
mask = mask(:,:,1);
mask = squeeze(mask);
mask = cast(mask,'logical');
save("data\mask_SZ_NM_N_F.mat", 'mask')