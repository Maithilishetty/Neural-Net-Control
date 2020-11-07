clc;close all;clear all;

a = 1; b = 1;
m0 = 2;
m1 = 20;
m2 = 10;
m3 = 1;
w1 = 0.2*rand(m1,m0+1);
w2 = 0.2*rand(m2,m1+1);
w3 = 0.2*rand(m3,m2+1);
eta1 = 0.03;
yp = [0;0];

for k=1:99998
if k<50000
        u = -1 + 2.*rand(1);
        p = [1;yp(end);yp(end-1)];
        y = yp(end)*yp(end-1)*(yp(end) + 2.5)/(1 + (yp(end))^2 + (yp(end-1))^2) + u;
        
        %FORWARD PASS
        v1 = w1*p;
        phi_v1 = a*tanh(b*v1); 
        y1_k = [1 ; phi_v1];
        v2 = w2*y1_k; 
        phi_v2 = a*tanh(b*v2);
        y2_k = [1 ; phi_v2];
        v3 = w3*y2_k; 
        y3 = v3 ;
        E = yp(end)*yp(end-1)*(yp(end) + 2.5)/(1 + (yp(end))^2 + (yp(end-1))^2) - y3;
        phi_v3_diff = 1;
        phi_v2_diff = a*(1 - phi_v2.^2); 
        phi_v1_diff = a*(1 - phi_v1.^2); 
   
        %BACKWARD PASS
        delta3 = E.*phi_v3_diff;  
        delta_w3 = eta1.*delta3*(y2_k)'; 
        delta2 = (w3(2:length(w3))'.*delta3).*(phi_v2_diff); 
        delta_w2 = eta1.*delta2*(y1_k)';
        delta1 = (w2(:,2:length(w2))'*delta2).*(phi_v1_diff);
        delta_w1 = eta1.*delta1*(p)';
        
        %ERROR CALCULATION AND WEIGHT UPDATION
        
        w1 = w1 + delta_w1;
        w2 = w2 + delta_w2;
        w3 = w3 + delta_w3;
        
        yp = [yp;y];
end
end

yp = [0;0];
ym = [0;0];

for k = 1:99998
    
        p = [1;yp(end);yp(end-1)];
        u = sin(2*pi*k/25);
        y1 = yp(end)*yp(end-1)*(yp(end) + 2.5)/(1 + (yp(end))^2 + (yp(end-1))^2) + u;
        
        %FORWARD PASS
        v1 = w1*p; 
        phi_v1 = a*tanh(b*v1); 
        y1_k = [1 ; phi_v1];
        v2 = w2*y1_k; 
        phi_v2 = a*tanh(b*v2);
        y2_k = [1 ; phi_v2];
        v3 = w3*y2_k;
        y2 = v3 + u;
        
        yp = [yp;y1];
        ym = [ym;y2];

end

% ----------------------------------------------------------------------- %
%                           Plotting Graphs                               %
% ----------------------------------------------------------------------- %

K = 1:100000;
figure;
plot(K,ym,'-r',K,yp,'--g')
legend('Neural Network Output', 'Desired Output');
title('Example 2 Function Approximation - Two Layer BPA');
axis([0 100 -4 4]);
g=var(yp(3:end)-ym(3:end));
h=var(yp(3:end));
perf=(1-(g/h))*100;
disp(perf)
mse = sum((yp(3:end)-ym(3:end)).^2)/(length(yp(3:end)));
disp(mse);