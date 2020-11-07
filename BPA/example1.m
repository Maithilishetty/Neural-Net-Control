clc;close all;clear all;
a = 1; b = 1;
m0 = 1;
m1 = 20;
m2 = 10;
m3 = 1;
w1 = 0.2*rand(m1,m0+1);%20 x 3
w2 = 0.2*rand(m2,m1+1);%10 x 21
w3 = 0.2*rand(m3,m2+1);%1 x 11
eta1 = 0.1;

for k=1:49998
    if (k<=10000)
    
        u = -1 + 2.*rand(1);
        p = [1;u];
          
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
        E = 0.6*sin(pi*u) + 0.3*sin(3*pi*u) + 0.1*sin(5*pi*u)- y3;
        phi_v3_diff = 1;
        phi_v2_diff = a*(1 - phi_v2.^2); %10 x 1
        phi_v1_diff = a*(1 - phi_v1.^2); % 20 x 1
   
        %BACKWARD PASS
    
        delta3 = E.*phi_v3_diff; % 1 x 1   
        delta_w3 = eta1.*delta3*(y2_k)'; % 1 x 11
        delta2 = (w3(2:length(w3))'.*delta3).*(phi_v2_diff); 
        delta_w2 = eta1.*delta2*(y1_k)';
        delta1 = (w2(:,2:length(w2))'*delta2).*(phi_v1_diff);
        delta_w1 = eta1.*delta1*(p)';
        
        %ERROR CALCULATION AND WEIGHT UPDATION
        
        w1 = w1 + delta_w1;
        w2 = w2 + delta_w2;
        w3 = w3 + delta_w3;
    end
end

yp = [0;0];
ym = [0;0];

for k = 1:1000
    
        u = sin(2*pi*k/250);
        p = [1;u];

        y1 = 0.3*yp(end) + 0.6*yp(end-1) +0.6*sin(pi*u) + 0.3*sin(3*pi*u) + 0.1*sin(5*pi*u);
        
        %FORWARD PASS
        v1 = w1*p;%20 x 1 
        phi_v1 = a*tanh(b*v1); %20 x 1
        y1_k = [1 ; phi_v1];%21 x 1
        v2 = w2*y1_k; %10 x 1
        phi_v2 = a*tanh(b*v2);
        %linear activation function
        y2_k = [1 ; phi_v2];%11 x 1
        v3 = w3*y2_k; %1 x 1
        y2 = (v3)+ 0.3*yp(end) + 0.6*yp(end-1);
        
        yp = [yp;y1];
        ym = [ym;y2];

end
K = 0:1001;
figure;
plot(K,ym,'-r',K,yp,'--g')
legend('Neural Network Output', 'Desired Output');
title('Example 1 Function Approximation - Two Layer BPA');
g=var(yp(3:end)-ym(3:end));
h=var(yp(3:end));
disp(g)
disp(h)
perf=(1-(g/h))*100;
disp(perf)
mse = sum((yp(3:end)-ym(3:end)).^2)/(length(yp(3:end)));
disp(mse);