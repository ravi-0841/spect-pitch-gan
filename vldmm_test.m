%% test matching

addpath(genpath('./'));
close all
% data = load('./data/cmu-arctic.mat');
sample = randperm(size(data.src_f0_feat,1));
sample = sample(1);
% src_feat = double(squeeze(data.src_f0_feat(sample,:,:)));
% tar_feat = double(squeeze(data.tar_f0_feat(sample,:,:)));
% momentum = double(squeeze(data.momenta_f0(sample,:,:)));

q = randperm(size(src_feat,1));
q = q(1);
x = 1:size(src_feat,2);
Z = [x' src_feat(q,:)'];
Y = [x' tar_feat(q,:)'];
% figure(1)
% hold on
% plot(Z(:,1),Z(:,2),'ro')
% plot(Y(:,1),Y(:,2),'bo')

disp([sample, q]);
[n,d] = size(Z);
defo.kernel_size_mom = [6, 50];
defo.nb_euler_steps = 15;

objfun.lambda = 1.5;

options.record_history =true;
options.maxit = 100;
options.nvec = 10;
options.prtlevel = 0;

P_op = momentum(q,:)';

% P = -rand(size(Z,1),1);
% [P_op,summary] = vmatch(Z,Y,P,defo,objfun,options);

[X_evol,P_evol] = forward_tan(Z,P_op,defo,1); 
[Xgr,Ygr] = ndgrid(0:5:130,0:20:400);
nxgrid = size(Xgr,1);
nygrid = size(Xgr,2);
gr = [Xgr(:) Ygr(:)];

[gr_evol] = flow(gr,Z(:,1),X_evol,P_evol,defo);

fig = figure(1);
    
for t=1:1:defo.nb_euler_steps+1
    [nx,d]=size(Z);
    clf;
    hold on;
    
    Xt=reshape(gr_evol{t}(:,1),size(Xgr));
    Yt=reshape(gr_evol{t}(:,2),size(Ygr));
    
    for x=1:nxgrid
        h=plot(Xt(x,:),Yt(x,:));
        set(h,'Color',.6*[1,1,1]);
    end
    
    for y=1:nygrid
        h=plot(Xt(:,y),Yt(:,y));
        set(h,'Color',.6*[1,1,1]);
    end

    xtemp = [Z(:,1) X_evol{t}]; 
    p1temp = [zeros(n,1),P_evol{t}]; 
   
    plot(xtemp(:,1),xtemp(:,2),'Marker','o','Color','r');
    plot(Y(:,1),Y(:,2),'Marker','o','Color','b');
    quiver(xtemp(:,1),xtemp(:,2),p1temp(:,1),p1temp(:,2),0,'Linewidth',1,'Color','g')
%    axis([-2,2,-2,2])
%    axis equal 
    frame = getframe(fig);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im,256);
    if t==1
        imwrite(imind, cm, '/home/ravi/Desktop/warping.gif', 'gif', ...
            'LoopCount', inf);
    else
        imwrite(imind, cm, '/home/ravi/Desktop/warping.gif', 'gif', ...
            'WriteMode', 'append');
    end
% axis off
% print(fig,['example',num2str(t)],'-dpng')
    pause(0.1)

end
