function [xp,yp,zp] = back_project(depth_map,K)

sz = size(depth_map);
[u,v]=meshgrid(1:sz(2),1:sz(1));
ons=ones(size(u));

C=[u(:)-1,v(:)-1];
u=C(:,1);v=C(:,2); r=depth_map(:);
mask0=abs(r)>=1;
u=u(mask0); v=v(mask0); ons=ons(mask0); r=r(mask0);
Kinv=pinv(K);
Xn=Kinv(1,1)*(u-1)+Kinv(1,2)*(v-1)+Kinv(1,3)*ons(:);
Yn=Kinv(2,1)*(u-1)+Kinv(2,2)*(v-1)+Kinv(2,3)*ons(:);
Zn=Kinv(3,1)*(u-1)+Kinv(3,2)*(v-1)+Kinv(3,3)*ons(:);
N=Zn(:);
xp = zeros(sz); yp = zeros(sz); zp = zeros(sz);
xp(mask0) = (Xn.*r)./N; yp(mask0) = (Yn.*r)./N; zp(mask0) = (Zn.*r)./N;
xp=reshape(xp,sz); 
yp=reshape(yp,sz);
zp=reshape(zp,sz);

