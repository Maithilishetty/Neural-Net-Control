clc;close all;clear all;
a = 1; b = 1;

m0 = 1;
m1 = 10;
m2 = 10;
m3 = 1;

w1 = rand(m1,m0+1);
w2 = rand(m2,m1+1);
w3 = rand(m3,m2+1);

w4 = rand(m1,m0+1);
w5 = rand(m2,m1+1);
w6 = rand(m3,m2+1);

eta = 0.015;
n = 0:1:10000;
yp = [0] ;
yphat = 0;

for k=1:length(n)
        
        u = sin(2*pi*k/25)+sin(2*pi*k/10);
        
        u1 = u^3;
        
        
        p = [1;yp(end)];
        
        

        %FORWARD PASS
        
        v1 = w1*p;
        phi_v1 = a*tanh(b*v1); 
        y1_k = [1 ; phi_v1];
        v2 = w2*y1_k; 
        phi_v2 = a*tanh(b*v2);
        y2_k = [1 ; phi_v2];
        v3 = w3*y2_k; 
        y3 = v3 ;
        
        %BACKWARD PASS
         
        
          
        p2 = [1;u];
        
        %FORWARD PASS
        
        v4= w4*p2;
        phi_v4 = a*tanh(b*v4); 
        y4_k = [1 ; phi_v4];
        v5 = w5*y4_k; 
        phi_v5 = a*tanh(b*v5);
        y5_k = [1 ; phi_v5];
        v6 = w6*y5_k; 
        y6= v6;
        
        %error calculation
        
        yp1 =(yp(end)/(1 + (yp(end))^2))+ u1;
        yp = [yp yp1];
        
        yphat1 = y3 + y6;
        yphat = [yphat yphat1];
        
        E = yp1 - yphat1;

        phi_v3_diff = 1;
        phi_v2_diff = (b/a)*(a^2 - phi_v2.^2); 
        phi_v1_diff = (b/a)*(a^2 - phi_v1.^2); 
        delta3 = E.*phi_v3_diff; 
        delta_w3 = eta.*delta3*(y2_k)'; 
        delta2 = (w3(2:length(w3))'.*delta3).*(phi_v2_diff); 
        delta_w2 = eta.*delta2*(y1_k)';
        delta1 = (w2(:,2:length(w2))'*delta2).*(phi_v1_diff);
        delta_w1 = eta.*delta1*(p)';
        
        %WEIGHT UPDATION
        
        w1 = w1 + delta_w1;
        w2 = w2 + delta_w2;
        w3 = w3 + delta_w3;
        phi_v6_diff = 1;
        phi_v5_diff = (a/b)*(a^2 - phi_v5.^2); %10 x 1
        phi_v4_diff = (a/b)*(a^2 - phi_v4.^2); % 20 x 1
   
        %BACKWARD PASS
    
        delta6 = E.*phi_v6_diff; % 1 x 1   
        delta_w6 = eta.*delta6*(y5_k)'; % 1 x 11
        delta5 = (w6(2:length(w6))'.*delta6).*(phi_v5_diff); 
        delta_w5 = eta.*delta5*(y4_k)';
        delta4 = (w5(:,2:length(w5))'*delta5).*(phi_v4_diff);
        delta_w4 = eta.*delta4*(p2)';
        
        % WEIGHT UPDATION
        w4 = w4 + delta_w4;
        w5 = w5 + delta_w5;
        w6 = w6 + delta_w6;
        

        

        
end


% ----------------------------------------------------------------------- %
%                           Plotting Graphs                               %
% ----------------------------------------------------------------------- %

K = 0:length(n);
figure;
plot(K,yp,'-r',K,yphat,'--g');
legend('Neural Network Output', 'Desired Output');
title('Example 3 Function Approximation - Two Layer BPA');
axis([0 500 -8 8])
g=var(yp(2:end)-yphat(2:end));
h=var(yp(2:end));
perf=(1-(g/h))*100;
disp(perf)
mse = sum((yp(2:end)-yphat(2:end)).^2)/(length(yp(2:end)));
disp(mse);