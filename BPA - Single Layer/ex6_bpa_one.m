clc;
close all;
clear all;
a = 2; b =1;
m0 = 1;
m1 = 20;
m2 = 1;
w1 = rand(m1,m0+1);%20 x 2
w2 = rand(m2,m1+1);%10 x 21
eta1=0.01;
yp=0;
yp_hat=0;

for k = 1:50000
    
   
   u_k=sin(2*pi*k/25);
   x_k = [1;u_k];
    v1 = w1*x_k;% 5*1
    
    %FORWARD PASS
    
    phi_v1 = a*tanh(b*v1);
    y1_k = [1 ; phi_v1];
    v2 = w2*y1_k;
    phi_v2 = v2; %linear activation function
    y2_k = v2;
    N_u(k)=y2_k;
    f_u(k)=(u_k-0.8).*u_k.*(u_k+0.5);
    yp=[yp;0.8*yp(end)+f_u(k)];
    yp_hat=[yp_hat;0.8*yp_hat(end)+N_u(k)];
    
    e_k=yp(end)-yp_hat(end);
        
    phi_v2_diff = 1;
    phi_v1_diff = a*(1-phi_v1.^2);
    
    %BACKWARD PASS
    
    delta2 = e_k.*phi_v2_diff;
    delta_w2 = eta1*delta2*(y1_k)';
    delta1 = (w2(2:length(w2))'.*delta2).*(phi_v1_diff);
    delta_w1 = eta1*delta1*(x_k)';
    
    %WEIGHT UPDATION
    
    w1 = w1+delta_w1;
    w2 = w2+delta_w2;
end

figure;
K=1:50001;
plot(K,yp_hat,'-r',K,yp,'--g');
legend('NN Output', 'desired Output');
title('Example 6 Function Approximation - Single Layer BPA');
axis([49500 50000 -2.5 0.5]);
g=var(yp(3:end)-yp_hat(3:end));
h=var(yp(3:end));
perf=(1-(g/h))*100;
disp(perf)
mse = sum((yp(3:end)-yp_hat(3:end)).^2)/(length(yp_hat(3:end)));
disp(mse);

