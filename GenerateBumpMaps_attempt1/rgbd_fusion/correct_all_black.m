function im = correct_all_black(image,mask)

R = image(:,:,1);
G = image(:,:,2);
B = image(:,:,3);
idx = R == 0 & G == 0 & B == 0 & mask == 1;
R(idx) = 1;
G(idx) = 1;
B(idx) = 1;
im = zeros(size(image),'uint8');
im(:,:,1) = R;
im(:,:,2) = G;
im(:,:,3) = B;
