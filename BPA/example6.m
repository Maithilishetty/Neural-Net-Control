clc;
close all;
clear all;

a = 2; b =1;
m0 = 1;
m1 = 20;
m2 = 10;
m3 = 1;
w1 = 0.1*rand(m1,m0+1);%20 x 2
w2 = 0.1*rand(m2,m1+1);%10 x 21
w3 = 0.1*rand(m3,m2+1);%1 x 11
eta1=0.01;
y=0;
yp=0;

for k = 1:1:50000
        u_k=sin(2*pi*k/25);
        p = [1;u_k];
        
        %FORWARD PASS
        v1 = w1*p;%20 x 1 
        phi_v1 = a*tanh(b*v1); %20 x 1
        y1_k = [1 ; phi_v1];%21 x 1
        v2 = w2*y1_k; %10 x 1
        phi_v2 = a*tanh(b*v2);
        %linear activation function
        y2_k = [1 ; phi_v2];%11 x 1
        v3 = w3*y2_k; %1 x 1
        y3 = v3 ;
        N_u(k)=y3;
        f_u(k)=(u_k-0.8).*u_k.*(u_k+0.5);
        y=[y;0.8*y(end)+f_u(k)];
        yp=[yp;0.8*yp(end)+y3];
        
        er=y(end)-yp(end);
        
        phi_v3_diff = (1);
        phi_v2_diff = (a/b)*(a^2 - phi_v2.^2); %10 x 1
        phi_v1_diff = (a/b)*(a^2 - phi_v1.^2); % 20 x 1
        %BACKWARD PASS
    
        delta3 = er.*phi_v3_diff; % 1 x 1   
        delta_w3 = eta1.*delta3*(y2_k)'; % 1 x 11
        delta2 = (w3(2:length(w3))'.*delta3).*(phi_v2_diff); 
        delta_w2 = eta1.*delta2*(y1_k)';
        delta1 = (w2(:,2:length(w2))'*delta2).*(phi_v1_diff);
        delta_w1 = eta1.*delta1*(p)';
      
        %WEIGHT UPDATION
        
        w1 = w1 + delta_w1;
        w2 = w2 + delta_w2;
        w3 = w3 + delta_w3;
         
        
end



figure;
K=1:50001;
plot(K,yp,'-r',K,y,'--g');
legend('NN Output', 'desired Output');
axis([49500 50000 -2.5 0.5]);
title('Example 6 Function Approximation - Two Layer BPA');

figure;
K=1:50000;
plot(K,N_u,'-r',K,f_u,'--g');
axis([49500 50000 -2.5 1]);
legend('NN Output', 'desired Output');

g=var(yp(3:end)-y(3:end));
h=var(yp(3:end));
perf=(1-(g/h))*100;
disp(perf)
mse = sum((yp(3:end)-y(3:end)).^2)/(length(yp(3:end)));
disp(mse);


% %Testing
% u=-1:0.1:1;
% f=(u-0.8).*u.*(u+0.5);
% length(u)
% for k = 1:length(u)
%         p = [1;u(k)];
%         
%         %FORWARD PASS
%         v1 = w1*p;%20 x 1 
%         phi_v1 = a*tanh(b*v1); %20 x 1
%         y1_k = [1 ; phi_v1];%21 x 1
%         v2 = w2*y1_k; %10 x 1
%         phi_v2 = a*tanh(b*v2);
%         %linear activation function
%         y2_k = [1 ; phi_v2];%11 x 1
%         v3 = w3*y2_k; %1 x 1
%         nn(k)=v3;
% end
% figure;
% size(u);
% size(nn);
% size(f);
% plot(u,nn,'-r',u,f,'--g');
% legend('Neural Network Output', 'Desired Output');
