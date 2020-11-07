% function ym1=neural_net_5a(yp1,yp2)
clc;close all;clear all;
a = 1; b = 1;
m0 = 2;
m1 = 20;
m2 = 1;
w11 = rand(m1,m0+1);%20 x 3
w21 = rand(m2,m1+1);%10 x 21

w12 = rand(m1,m0+1);%20 x 3
w22 = rand(m2,m1+1);%10 x 21

eta1 = 0.1;

yp1= [0];
yp2= [0];
yp1_hat=0;
yp2_hat=0;

for k=1:10000

    
%         u1 = -1 + 2.*rand(1);
%         u2 = -1 + 2.*rand(1);
         u1 = sin(2*pi*k/25);
         u2 = cos(2*pi*k/25);

        p = [1;yp1(end);yp2(end)];
      
        y1 = yp1(end)/(1 + (yp2(end))^2) + u1;
        y2 =(yp1(end)*yp2(end))/(1 + (yp2(end))^2) + u2 ;
        
         yp1=[yp1;y1];
         yp2=[yp2;y2];

        %FORWARD PASS
        v11 = w11*p;%20 x 1 
        phi_v11 = a*tanh(b*v11); %20 x 1
        y11_k = [1 ; phi_v11];%21 x 1
        v21 = w21*y11_k; %10 x 1
        
        y31 = v21 ;
        
        yp1_hat=[yp1_hat;y31+u1];
        
        E1 = yp1(end) - yp1_hat(end);
        phi_v21_diff = (1);
        phi_v11_diff = a*(1 - phi_v11.^2); % 20 x 1
   
        %BACKWARD PASS
    
        delta21 = E1.*phi_v21_diff; % 1 x 1   
        delta_w21 = eta1.*delta21*(y11_k)'; % 1 x 11
        
        delta11 = (w21(:,2:length(w21))'*delta21).*(phi_v11_diff);
        delta_w11 = eta1.*delta11*(p)';
        
        %ERROR CALCULATION AND WEIGHT UPDATION
         
        w11 = w11 + delta_w11;
        w21 = w21 + delta_w21;
        
        %FORWARD PASS
        v12 = w12*p;%20 x 1 
        phi_v12 = a*tanh(b*v12); %20 x 1
        y12_k = [1 ; phi_v12];%21 x 1
        v22 = w22*y12_k; %10 x 1
        
        y32 = v22 ;
        yp2_hat=[yp2_hat;y32+u2];
        E2=yp2(end) - yp2_hat(end); 
        phi_v22_diff = (1);
        phi_v12_diff = a*(1 - phi_v12.^2); % 20 x 1
   
        %BACKWARD PASS
    
        delta22 = E2.*phi_v22_diff; % 1 x 1   
        delta_w22 = eta1.*delta22*(y12_k)'; % 1 x 11
       
        delta12 = (w22(:,2:length(w22))'*delta22).*(phi_v12_diff);
        delta_w12 = eta1.*delta12*(p)';
        
        %ERROR CALCULATION AND WEIGHT UPDATION
         
        w12 = w12 + delta_w12;
        w22 = w22 + delta_w22;
        
        
end
K=1:1:k+1;
figure;
plot(K,yp1_hat,'-r',K,yp1,'--g')
legend('Neural Network Output', 'Desired Output');
title('Example 5 Function Approximation for yp1- Single Layer BPA');
axis([0 500 -5 5]);

figure;
plot(K,yp2_hat,'-r',K,yp2,'--g')
legend('Neural Network Output', 'Desired Output');
title('Example 5 Function Approximation for yp2 - Single Layer BPA');
axis([9900 10000 -5 5]);

g=var(yp1(2:end)-yp1_hat(2:end));
h=var(yp1(2:end));
perf=(1-(g/h))*100;
disp(perf)
mse = sum((yp1(2:end)-yp1_hat(2:end)).^2)/(length(yp1(2:end)));
disp(mse);

g=var(yp2(2:end)-yp2_hat(2:end));
h=var(yp2(2:end));
perf=(1-(g/h))*100;
disp(perf)
mse = sum((yp2(2:end)-yp2_hat(2:end)).^2)/(length(yp2(2:end)));
disp(mse);

